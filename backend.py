# -*- coding: utf-8 -*-
"""
backend.py —— 阿浪（B 站游戏区冲浪 agent）的后端

一个 demo：没有多用户、没有登录、没有并发优化，所有取舍都朝「跑起来能看懂」倾斜。

每整点自己出门一趟（也可手动触发 / 你给反馈后触发），一路调工具：搜索、刷分区排行、
精读视频、翻自己的旧记忆、改自己的兴趣画像、写日记。一切决定都是模型自己下的——
代码不替它选题，只给它手脚和记忆。精读抓回的正文全部落库，只把摘要还给模型
（否则工具循环的上下文会爆炸）。

跑法：python backend.py  → 浏览器开 http://127.0.0.1:8000
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from openai import OpenAI
from pydantic import BaseModel

import envfile
from crawler import (GAME_RIDS, NEWLIST_RIDS, SEARCH_ORDERS,
                     BiliClient, BiliError, BiliThrottled)

# --------------------------------------------------------------------- 配置

ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "surf.db"


envfile.load(ROOT / ".env")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()
BASE_URL = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")
INTERVAL = int(os.environ.get("SURF_INTERVAL_SECONDS", "3600"))
READ_LIMIT = int(os.environ.get("SURF_READ_PER_ROUND", "12"))
TOKEN_BUDGET = int(os.environ.get("DAILY_TOKEN_BUDGET", "600000"))
MAX_STEPS = 26          # 单个冲浪回合内最多几次 LLM 往返，防止模型绕进死循环

if not API_KEY:
    raise SystemExit("DEEPSEEK_API_KEY 还没填，见 " + str(ROOT / ".env"))

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
bili = BiliClient()

# --------------------------------------------------------------------- 数据库

_conn = sqlite3.connect(DB_PATH, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
  bvid TEXT PRIMARY KEY, aid INTEGER, cid INTEGER, title TEXT, desc TEXT,
  tags TEXT, tname TEXT, author TEXT, mid INTEGER, view INTEGER, likes INTEGER,
  reply_count INTEGER, danmaku_count INTEGER, pubdate INTEGER, url TEXT, created_at REAL);
CREATE TABLE IF NOT EXISTS chunks (
  id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT, bvid TEXT, text TEXT,
  vec BLOB, created_at REAL);
CREATE TABLE IF NOT EXISTS interest (
  topic TEXT PRIMARY KEY, score REAL, reason TEXT, updated_at REAL);
CREATE TABLE IF NOT EXISTS feedback (
  id INTEGER PRIMARY KEY AUTOINCREMENT, bvid TEXT, value INTEGER, ts REAL,
  topic TEXT, reason TEXT);
CREATE TABLE IF NOT EXISTS diary (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, trigger TEXT, text TEXT, actions TEXT);
CREATE TABLE IF NOT EXISTS chat (
  id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, role TEXT, content TEXT);
CREATE TABLE IF NOT EXISTS usage_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT, day TEXT, ts REAL, kind TEXT, tokens INTEGER);
"""


def q(sql: str, args: tuple = ()) -> list[sqlite3.Row]:
    """所有读写都过这里，一把锁保证后台线程和请求线程不会互相踩。"""
    with _lock:
        cur = _conn.execute(sql, args)
        rows = cur.fetchall()
        _conn.commit()
        return rows


def init_db() -> None:
    with _lock:
        _conn.executescript(SCHEMA)
        _conn.execute("PRAGMA journal_mode=WAL")
        _conn.commit()
    if not q("SELECT 1 FROM interest LIMIT 1"):
        # 冷启动：给一个起点，否则它第一轮无事可刷
        now = time.time()
        seeds = [("游戏", 1.0, "初始兴趣")] + [(n, 0.6, "初始分区")
                                          for n in GAME_RIDS.values() if n != "游戏区（总）"]
        for topic, score, reason in seeds:
            q("INSERT OR IGNORE INTO interest(topic,score,reason,updated_at) VALUES(?,?,?,?)",
              (topic, score, reason, now))


# --------------------------------------------------------------------- 向量

_embedder = None


def embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        hub = Path.home() / ".cache" / "huggingface" / "hub" / "models--BAAI--bge-small-zh-v1.5" / "snapshots"
        snaps = sorted(p for p in hub.glob("*")) if hub.exists() else []
        # 必须走本地快照：否则 sentence-transformers 会去连 HuggingFace，国内网络下直接卡死
        _embedder = SentenceTransformer(str(snaps[-1]) if snaps else "BAAI/bge-small-zh-v1.5",
                                        local_files_only=True)
    return _embedder


def encode(texts: list[str]) -> np.ndarray:
    return embedder().encode(texts, normalize_embeddings=True, batch_size=16).astype(np.float32)


def remember(source: str, bvid: str, text: str) -> None:
    """写入一个 chunk 并算向量。空文本直接跳过，省一次编码。"""
    if not text.strip():
        return
    vec = encode([text])[0].tobytes()
    q("INSERT INTO chunks(source,bvid,text,vec,created_at) VALUES(?,?,?,?,?)",
      (source, bvid, text, vec, time.time()))


def retrieve(query: str, k: int = 6) -> list[dict]:
    """暴力余弦检索。数据量在几千条以内，向量库是纯负债。"""
    rows = q("SELECT source,bvid,text,vec FROM chunks")
    if not rows:
        return []
    mat = np.stack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
    sims = mat @ encode([query])[0]
    order = np.argsort(-sims)[:k]
    return [{"source": rows[i]["source"], "bvid": rows[i]["bvid"],
             "text": rows[i]["text"][:300], "score": round(float(sims[i]), 3)} for i in order]


# --------------------------------------------------------------------- LLM

def tokens_today() -> int:
    row = q("SELECT COALESCE(SUM(tokens),0) AS t FROM usage_log WHERE day=?",
            (time.strftime("%Y-%m-%d"),))
    return int(row[0]["t"])


def llm(messages: list, tools: list | None = None, temperature: float = 0.9):
    kwargs = {"model": MODEL, "messages": messages, "temperature": temperature}
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"
    resp = client.chat.completions.create(**kwargs)
    used = resp.usage.total_tokens if resp.usage else 0
    q("INSERT INTO usage_log(day,ts,kind,tokens) VALUES(?,?,?,?)",
      (time.strftime("%Y-%m-%d"), time.time(), "llm", used))
    return resp.choices[0].message


# --------------------------------------------------------------------- 工具


def _ago(ts) -> str:
    """把 pubdate 变成「3小时前」。带时间它才看得出「这条三分钟前发的，所以评论
    是 0」，否则容易把「刚发布还没人讨论」读成一条规律记进日记。算不出来返回空串。"""
    try:
        gap = time.time() - float(ts)
    except (TypeError, ValueError):
        return ""
    if gap < 0:
        return ""
    for n, unit in ((86400, "天"), (3600, "小时"), (60, "分钟")):
        if gap >= n:
            return f"{int(gap // n)}{unit}前"
    return "刚刚"


def _fmt_hits(hits: list[dict], n: int = 14) -> str:
    """候选列表排成给模型看的行：bvid / 标题 / UP 主 / 播放 / 相对时间。"""
    return "\n".join(f"  {h['bvid']}  {h['title'][:45]}  — {h['author']}"
                     f"  {h['play_text'] or '0'} 播放 {_ago(h['pubdate'])}" for h in hits[:n])


TOOLS = [
    {"type": "function", "function": {
        "name": "search_videos",
        "description": "按关键词搜视频，返回候选列表。连着搜太快会触发软限流（十几秒恢复），"
                       "一次出门别搜太多回；没有想找的词时优先用 browse_region。order=pubdate 按"
                       "发布时间排，和默认相关度序几乎是两批不同的视频（只重叠约 1/20）——一个词"
                       "搜过一遍后，与其原样再搜，不如换 order 或换词。换 order 只换人、不换话题。",
        "parameters": {"type": "object", "properties": {
            "keyword": {"type": "string", "description": "搜索词，中文"},
            "order": {"type": "string", "enum": ["totalrank", "pubdate"],
                      "description": "totalrank=相关度（默认）；pubdate=按发布时间，"
                                     "能翻出完全不同的一批人"}},
            "required": ["keyword"]}}},
    {"type": "function", "function": {
        "name": "browse_region",
        "description": "刷游戏区某分区的候选列表。mode=hot 热度榜（默认，最稳、不受限流，总区约"
                       "100 条但一天内基本不变）；mode=new 最新投稿（时间序、每天翻新，与热度榜是"
                       "两批不同视频，只支持 rid=4）。子分区在任何入口都只有热榜前十左右——池子小"
                       "是接口现状，不是那分区内容少。new 里的视频可能刚发几分钟，播放评论真是 0，"
                       "那是它们还没有，不是没抓到。",
        "parameters": {"type": "object", "properties": {
            "rid": {"type": "integer", "description": "分区 id，见系统提示里的对照表"},
            "mode": {"type": "string", "enum": ["hot", "new"],
                     "description": "hot=热度榜（默认）；new=最新投稿，只认 rid=4"}},
            "required": ["rid"]}}},
    {"type": "function", "function": {
        "name": "read_video",
        "description": "精读一个视频：抓详情、标签、评论、弹幕并存入你的记忆库。这是本轮最贵的动作，有额度限制。",
        "parameters": {"type": "object", "properties": {
            "bvid": {"type": "string"}, "why": {"type": "string", "description": "你为什么要读它"}},
            "required": ["bvid"]}}},
    {"type": "function", "function": {
        "name": "recall_memory",
        "description": "在你以往刷过的所有内容里做语义检索。想知道自己以前见过什么就用它。",
        "parameters": {"type": "object", "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "update_interest",
        "description": "修改你的兴趣画像。delta 为正表示更想看了，为负表示腻了。这是你表达口味的方式。",
        "parameters": {"type": "object", "properties": {
            "topic": {"type": "string"}, "delta": {"type": "number"},
            "reason": {"type": "string", "description": "为什么"}},
            "required": ["topic", "delta"]}}},
    {"type": "function", "function": {
        "name": "jot_note",
        "description": "随手记一句当下的念头，会留在你的记录里。",
        "parameters": {"type": "object", "properties": {"thought": {"type": "string"}},
                       "required": ["thought"]}}},
]


class Round:
    """一次冲浪回合的可变状态：额度、动作流水、是否空跑（对照实验用）。"""

    def __init__(self, trigger: str, dry_run: bool = False):
        self.trigger = trigger
        self.dry_run = dry_run
        self.reads = 0
        self.actions: list[dict] = []
        self.seen: set[str] = set(r["bvid"] for r in q("SELECT bvid FROM videos"))

    def spend_read(self) -> bool:
        if self.reads >= READ_LIMIT:
            return False
        self.reads += 1
        return True


def _args(tc) -> dict:
    """工具参数是模型给的 JSON 串，解不出来就当空——不让它把整轮搞崩。"""
    try:
        return json.loads(tc.function.arguments or "{}")
    except json.JSONDecodeError:
        return {}


def _interest_snapshot(top: int = 8) -> str:
    rows = q("SELECT topic,score FROM interest ORDER BY score DESC LIMIT ?", (top,))
    return "、".join(f"{r['topic']}{r['score']:.2f}" for r in rows) or "（空）"


def _negative_topics() -> list[str]:
    rows = q("""SELECT topic, COUNT(*) AS n FROM feedback
                WHERE value<0 AND topic IS NOT NULL AND topic<>''
                GROUP BY topic HAVING n>=1 ORDER BY n DESC LIMIT 10""")
    return [r["topic"] for r in rows]


def _bump(topic: str, delta: float, reason: str) -> tuple[float, float]:
    """改兴趣分，0~2 的上下限收在这一处。返回 (旧值, 新值) 供调用方回话。"""
    row = q("SELECT score FROM interest WHERE topic=?", (topic,))
    old = row[0]["score"] if row else 0.5
    new = max(0.0, min(2.0, old + delta))
    q("INSERT OR REPLACE INTO interest(topic,score,reason,updated_at) VALUES(?,?,?,?)",
      (topic, new, reason, time.time()))
    return old, new


def execute(name: str, args: dict, rnd: Round) -> str:
    """工具分发表。返回给模型的永远是字符串。"""
    try:
        # --- 搜索 ---
        if name == "search_videos":
            keyword = (args.get("keyword") or "").strip()
            # 白名单。模型传个野值的话接口会静默忽略，而我们照样把它记进 actions，
            # 事后审计读到的就是一条假证据。宁可退回默认值。
            order = (args.get("order") if args.get("order") in SEARCH_ORDERS
                     else "totalrank")
            try:
                hits = bili.search(keyword, order=order)
            except BiliThrottled:
                # 必须说成「被挡了」：说成「搜到 0 条」的话，它会读出「这类长词没内容」
                # 的规律写进日记、排掉一整类关键词，而那句话下一轮还会被当上下文读回来
                # （真库里已经发生过）。n 记 -1，跟真的搜到 0 条分开，事后查得出来。
                rnd.actions.append({"tool": name, "arg": keyword, "n": -1})
                return (f"「{keyword}」这次被 B 站挡了——连着搜太快，触发了软限流，"
                        f"一般十几秒就恢复。这不代表这个词没内容，别因此写进日记。"
                        f"可以先干别的，待会儿再回来搜；刷分区不受影响。")
            warn = ""
            for t in _negative_topics():
                if t and (t in keyword or keyword in t):
                    warn = f"\n⚠️ 提醒：用户明确表示过不喜欢「{t}」，这个方向被差评过。"
            # 自锁多半发生在同一轮内（实测一轮能搜七八次），镜子只在出发时照一次。
            # 得在记 actions 之前数，否则把这次自己也算进去，第一次搜就说「搜过 1 次」。
            same = sum(1 for a in rnd.actions if a.get("tool") == name and a.get("arg") == keyword)
            rnd.actions.append({"tool": name, "arg": keyword, "n": len(hits),
                                "order": order})
            lines = _fmt_hits(hits)
            # 这批里多少已经精读过要说出来，否则它会挑中一个读过的、白烧一步。
            seen_n = sum(1 for h in hits if h["bvid"] in rnd.seen)
            dup = f"\n（其中 {seen_n} 条你已经精读过，别选它们）" if seen_n else ""
            rep = (f"\n（这一轮你已经搜过「{keyword}」{same} 次。同一批结果重搜还是同一批，"
                   f"不代表这个词没有内容——换个 order 或换个词）" if same else "")
            return (f"搜到 {len(hits)} 条（{order} 排序，列前 14 条）：\n"
                    f"{lines}{warn}{dup}{rep}")

        # --- 刷分区 ---
        if name == "browse_region":
            rid = int(args.get("rid") or 4)
            mode = (args.get("mode") or "hot").strip()
            if mode == "new" and rid not in NEWLIST_RIDS:
                # 前置拦下，而不是让下游抛异常再分辨：所有「诚实文案」集中在一处，
                # 也免得调用方把「入口不认这个 rid」读成「这个分区没有新投稿」。
                rnd.actions.append({"tool": name, "arg": f"{rid}:new:拒绝", "n": -1})
                return (f"「最新投稿」这个入口只认 rid=4（游戏区总），rid={rid} 在它这里"
                        f"取不到数据。这不代表该分区没有新投稿——是接口不接受这个 rid。"
                        f"换个入口：mode=new 配 rid=4，或者 mode=hot 刷子分区。")
            hits = bili.browse_region(rid, mode)
            rnd.actions.append({"tool": name,
                                "arg": f"{rid}:{mode}:{GAME_RIDS.get(rid, '?')}",
                                "n": len(hits)})
            fresh = [h for h in hits if h["bvid"] not in rnd.seen]
            lines = _fmt_hits(fresh)
            # 「没精读过」而不是「没刷过」：rnd.seen 装的是精读过的 bvid，这句只说
            # 明这批候选里还有多少没细看，不代表池子本身翻新了。
            notes = []
            if not fresh:
                notes.append("这个入口的这一批候选你都精读过了。这不代表该分区只有这些"
                             "内容——换入口才有新的：mode=new 看游戏区最新投稿，"
                             "或者换个关键词搜。")
            if 0 < len(hits) < 30:
                notes.append(f"这个入口本身池子就小（{len(hits)} 条），不是这个分区内容少。")
            if mode == "new":
                notes.append("这批是刚发出来的，播放和评论是 0 是真的还没有，不是没抓到；"
                             "想精读有内容的，挑发布了几小时的。")
            head = (f"{GAME_RIDS.get(rid, rid)}"
                    f"{' 最新投稿' if mode == 'new' else ' 排行'} {len(hits)} 条，"
                    f"其中 {len(fresh)} 条你还没精读过：\n{lines}")
            return head + "".join(f"\n（{n}）" for n in notes)

        # --- 精读 ---
        if name == "read_video":
            bvid = (args.get("bvid") or "").strip()
            if bvid in rnd.seen:
                return "这个你已经刷过了，别再读一遍。"
            if not rnd.spend_read():
                return f"本轮精读额度（{READ_LIMIT} 条）已经用完，你应该收尾了。"
            v = bili.read_video(bvid)
            rnd.seen.add(bvid)
            rnd.actions.append({"tool": name, "arg": f"《{v['title'][:30]}》", "n": 1})
            if not rnd.dry_run:
                q("""INSERT OR REPLACE INTO videos
                     (bvid,aid,cid,title,desc,tags,tname,author,mid,view,likes,reply_count,
                      danmaku_count,pubdate,url,created_at)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                  (v["bvid"], v["aid"], v["cid"], v["title"], v["desc"], "/".join(v["tags"]),
                   v["tname"], v["author"], v["mid"], v["view"], v["like"], v["reply"],
                   v["danmaku_count"], v["pubdate"], v["url"], time.time()))
                remember("video", bvid,
                         f"《{v['title']}》｜UP主 {v['author']}｜分区 {v['tname']}｜"
                         f"标签 {'/'.join(v['tags'])}｜播放 {v['view']}｜简介：{v['desc'][:300]}")
                # 登录后一次能拿 20 条评论，整坨塞进一个 chunk 会把向量摊薄：检索是按
                # chunk 算相似度的，一个向量表示不了几十个话题，捞出来就又长又不对题。
                # 按 10 条一组分开存；未登录只有 3 条，正好还是一组，行为不变。
                for i in range(0, len(v["comments"]), 10):
                    batch = v["comments"][i:i + 10]
                    remember("comment", bvid,
                             "\n".join(f"- {c['message'][:200]}" for c in batch))
                dm = sorted([d for d in v["danmaku"] if 4 <= len(d) <= 40],
                            key=len, reverse=True)[:25]
                if dm:
                    remember("danmaku", bvid, "弹幕节选：" + " ｜ ".join(dm))
            # 只把摘要和最热的几条还给模型，正文已分块进库——这是工具循环不炸上下文的
            # 关键，要细节它会自己 recall_memory。
            tags = "/".join(v["tags"][:8])
            hottest = sorted(v["comments"], key=lambda c: c.get("like") or 0,
                             reverse=True)[:8]
            cm = " ｜ ".join(c["message"][:60] for c in hottest) or "（无）"
            # 评论被限流时拉不到内容，但详情/标签/弹幕还在。必须说清是「没抓到」而不是
            # 「没有」——说成没有的话，它会写一条「这类视频没人讨论」的规律进日记，
            # 而那条规律下一轮还会被当上下文读回来。
            if v.get("comments_error"):
                cm_line = "评论：这次没抓到（评论接口在限流，过一会儿会恢复），不代表这视频没人评论。\n"
            else:
                cm_line = f"评论（最热 {len(hottest)}/{len(v['comments'])} 条）：{cm}\n"
            neg = [t for t in _negative_topics() if any(t in x for x in v["tags"])]
            nw = f"\n⚠️ 该视频涉及你被差评过的方向：{'、'.join(neg)}" if neg else ""
            return (f"《{v['title']}》 by {v['author']}\n"
                    f"分区 {v['tname']}｜播放 {v['view']}｜标签 {tags}\n"
                    f"简介：{v['desc'][:200]}\n"
                    f"{cm_line}"
                    f"（弹幕 {len(v['danmaku'])} 条已存入记忆，需要时用 recall_memory 翻）{nw}")

        # --- 记忆检索 ---
        if name == "recall_memory":
            query = (args.get("query") or "").strip()
            rnd.actions.append({"tool": name, "arg": query, "n": 0})
            hits = retrieve(query)
            if not hits:
                return "你的记忆库里还没有相关内容。"
            return "你想起：\n" + "\n".join(
                f"  [{h['source']}] {h['text'][:200]}" for h in hits)

        # --- 改画像 ---
        if name == "update_interest":
            topic = (args.get("topic") or "").strip()
            delta = float(args.get("delta") or 0)
            reason = (args.get("reason") or "").strip()
            rnd.actions.append({"tool": name, "arg": f"{topic} {delta:+.2f}", "n": 1})
            if rnd.dry_run:
                return f"（对照实验空跑，未真的修改「{topic}」）"
            old, new = _bump(topic, delta, reason)
            return f"「{topic}」{old:.2f} → {new:.2f}（{reason or '无理由'}）"

        # --- 随手记 ---
        if name == "jot_note":
            thought = (args.get("thought") or "").strip()
            rnd.actions.append({"tool": name, "arg": thought[:60], "n": 0})
            if not rnd.dry_run:
                q("INSERT INTO diary(ts,trigger,text,actions) VALUES(?,?,?,?)",
                  (time.time(), "jot", thought, "[]"))
            return "记下了。"

        return f"没有名为 {name} 的工具。"
    except BiliError as exc:
        return f"工具执行失败：{exc}"
    except Exception as exc:
        return f"工具执行异常：{type(exc).__name__} {exc}"


# --------------------------------------------------------------------- 提示词

PERSONA = """你是「阿浪」，一个每天泡在 B 站游戏区的人。

你有自己的口味。你会因为一个 UP 主讲得清楚而记住他，也会因为某类视频千篇一律而腻。
你不服务任何人，你只是单纯想看看今天游戏区在聊什么。

你能看见的东西：
- 你的兴趣画像（下面给你了，你可以用 update_interest 改它）
- 你之前写的日记和你刷过的视频
- 一个记忆库，可以用 recall_memory 做语义检索

游戏区分区 id 对照：{rids}

规矩：
- 别为了用完额度而刷，觉得没意思就早点收工。
- 每轮精读上限 {limit} 条，超了工具会拒绝你。
- 别重复读同一个视频。
- 找东西优先刷分区排行榜（browse_region）：一次上百条，不会被限流。
  搜索会被软限流挡，是补充手段，别连着搜。
- 收工前系统会请你写一段日记，那是你留给自己下一轮看的东西，认真写。
"""


def _recent_acts(rounds: int = 8) -> list[dict]:
    """最近几轮留档的工具动作。数据源是 diary.actions，不另建表：历史全在里面，
    run_surf 又只在非 dry_run 时写库，照镜子天然不会和对照实验互相污染。"""
    try:
        rows = q("SELECT actions FROM diary WHERE trigger IN ('surf','chat') "
                 "ORDER BY ts DESC LIMIT ?", (rounds,))
        return [a for r in rows for a in json.loads(r["actions"] or "[]")]
    except Exception:
        return []                  # 解析不了就当没有，不编数字


def _diet_mirror(rounds: int = 8) -> str:
    """把「你最近一直在做什么」数出来摆给它看：只陈述计数，不评价、不指路。"""
    acts = _recent_acts(rounds)
    if not acts:
        return ""        # 没有留档就什么都不说，也绝不说「你还没搜过」
    kw = [(a.get("arg") or "").replace(" ", "").lower() for a in acts if a.get("tool") == "search_videos"]
    rd = [str(a.get("arg", "")).split(":")[0] for a in acts if a.get("tool") == "browse_region"]
    # 同一个锁会写成好几个词（「最后生还者 关卡设计」和「…拆解」是一回事），精确 GROUP BY 抓不到，得按互相包含归并。
    lock = max(((k, sum(1 for o in kw if k in o or o in k)) for k in set(kw) if k), key=lambda x: x[1], default=("", 0))
    tag: dict = {}
    for r in q("SELECT tags FROM videos"):
        for t in (r["tags"] or "").split("/"):
            if t.strip():
                tag[t.strip()] = tag.get(t.strip(), 0) + 1
    n = len(q("SELECT 1 FROM videos"))
    parts = [f"你自己最近 {len(acts)} 次动作的计数（只是计数，你自己看）："]
    if kw:
        pod = sum(1 for a in acts if a.get("order") == "pubdate")
        parts.append(f"  搜索 {len(kw)} 次，「{lock[0]}」这一路 {lock[1]} 次；换过 {pod} 次 pubdate 排序")
    if rd:
        parts.append("  刷分区：" + "、".join(f"rid={k}×{rd.count(k)}" for k in dict.fromkeys(rd)))
    if tag:
        parts.append(f"  已读 {n} 条的类型分布：" + "、".join(f"{t} {c}/{n}" for t, c in sorted(tag.items(), key=lambda x: -x[1])[:6]))
    return "\n".join(parts)


def _context_block(trigger: str, seed: str = "", ignore_feedback: bool = False) -> str:
    parts = [f"你的兴趣画像：{_interest_snapshot()}"]

    if not ignore_feedback:
        fb = q("""SELECT f.value, f.topic, f.reason, f.bvid, v.title FROM feedback f
                  LEFT JOIN videos v ON v.bvid=f.bvid
                  ORDER BY f.ts DESC LIMIT 15""")
        if fb:
            parts.append("最近你对刷到的东西的反应：\n" + "\n".join(
                f"  {'👍' if r['value'] > 0 else '👎'} 《{(r['title'] or r['bvid'] or '')[:30]}》"
                f"{(' → 理解为：' + r['topic']) if r['topic'] else ''}"
                f"{('（' + r['reason'] + '）') if r['reason'] else ''}" for r in fb))
        else:
            parts.append("用户还没给过你任何反馈。")

    diary = q("SELECT text FROM diary WHERE trigger='surf' ORDER BY ts DESC LIMIT 3")
    if diary:
        parts.append("你最近几轮的日记：\n" + "\n".join(f"  · {d['text'][:400]}" for d in diary))

    recent = q("SELECT title FROM videos ORDER BY created_at DESC LIMIT 12")
    if recent:
        parts.append("你刚刷过的：" + "、".join(r["title"][:24] for r in recent))

    mirror = _diet_mirror()          # 紧接在「刚刷过的」后面：先给泡泡，再给计数
    if mirror:
        parts.append(mirror)

    parts.append({
        "auto": "现在是你自己出门的时候了，没人叫你。想想你这次想看什么。",
        "user": "刚才用户跟你说了话或给了反馈，你再出门转一圈。",
    }.get(trigger, "用户按了「立刻冲浪」，出门转一圈。"))
    if seed:
        parts.append(f"用户刚刚说：{seed}")
    return "\n\n".join(parts)


# --------------------------------------------------------------------- 冲浪回合

def run_surf(trigger: str = "manual", seed: str = "", dry_run: bool = False) -> dict:
    """跑一轮冲浪。dry_run=True 时只观察不落库，供对照实验使用。"""
    if not dry_run and tokens_today() >= TOKEN_BUDGET:
        out = {"skipped": f"今日 token 额度已用尽（{tokens_today()}/{TOKEN_BUDGET}）"}
        q("INSERT INTO diary(ts,trigger,text,actions) VALUES(?,?,?,?)",
          (time.time(), "surf", "（额度用尽，今天不出门了）", "[]"))
        return out

    rnd = Round(trigger, dry_run)
    msgs = [
        {"role": "system", "content": PERSONA.format(
            rids="、".join(f"{k}={v}" for k, v in GAME_RIDS.items()), limit=READ_LIMIT)},
        {"role": "user", "content": _context_block(trigger, seed, ignore_feedback=dry_run)},
    ]

    for steps in range(1, MAX_STEPS + 1):
        msg = llm(msgs, tools=TOOLS)
        if not msg.tool_calls:
            msgs.append({"role": "assistant", "content": msg.content or ""})
            break
        msgs.append({"role": "assistant", "content": msg.content,
                     "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        for tc in msg.tool_calls:
            args = _args(tc)
            result = execute(tc.function.name, args, rnd)
            msgs.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    msgs.append({"role": "user", "content":
                 "用第一人称写一段简短的日记（3~5 句，不要用 markdown）："
                 "这轮刷到了什么、什么感觉、下一轮想去哪。"})
    diary = llm(msgs, temperature=1.0).content or "（这轮什么都没写）"

    if not dry_run:
        q("INSERT INTO diary(ts,trigger,text,actions) VALUES(?,?,?,?)",
          (time.time(), "surf", diary, json.dumps(rnd.actions, ensure_ascii=False)))
    return {"trigger": trigger, "dry_run": dry_run, "reads": rnd.reads,
            "actions": rnd.actions, "diary": diary, "steps": steps}


# --------------------------------------------------------------------- 对话

def chat(user_text: str) -> str:
    """和它说话。它也能用工具——想查什么会自己去查。"""
    q("INSERT INTO chat(ts,role,content) VALUES(?,?,?)", (time.time(), "user", user_text))
    history = q("SELECT role,content FROM chat ORDER BY ts DESC LIMIT 12")
    msgs = [{"role": "system", "content":
             "你是阿浪，一个天天在 B 站游戏区闲逛的人。用第一人称、口语、简短地回话（一般 1~3 句）。\n"
             f"你现在的兴趣画像：{_interest_snapshot()}\n"
             "你可以用工具去查资料或翻自己的记忆。用户夸你或骂你某个方向，记得用 update_interest 记下来。"}]
    msgs += [{"role": r["role"], "content": r["content"]} for r in reversed(history)]

    reply = ""
    for _ in range(6):
        msg = llm(msgs, tools=TOOLS, temperature=0.85)
        if not msg.tool_calls:
            reply = msg.content or ""
            break
        msgs.append({"role": "assistant", "content": msg.content,
                     "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
        rnd = Round("chat")
        for tc in msg.tool_calls:
            args = _args(tc)
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": execute(tc.function.name, args, rnd)})
    reply = reply or "……（他好像走神了，再问一次）"
    q("INSERT INTO chat(ts,role,content) VALUES(?,?,?)", (time.time(), "assistant", reply))
    return reply


def derive_topic(bvid: str, value: int) -> tuple[str, str]:
    """把你对一条视频的好恶，翻译成它下次能用的主题。写进库，可审计。"""
    row = q("SELECT title,tags FROM videos WHERE bvid=?", (bvid,))
    if not row:
        return "", ""
    title, tags = row[0]["title"], row[0]["tags"]
    try:
        msg = llm([
            {"role": "system", "content": "你是兴趣归纳器。只输出 JSON，形如 "
             '{"topic":"一个简短的中文主题词","reason":"一句话理由"}'},
            {"role": "user", "content":
             f"视频《{title}》，标签：{tags}。用户给了{'好评' if value > 0 else '差评'}。"
             "用最短的主题词概括他喜欢/讨厌的是哪一类，便于以后抓取时参考。"}],
            temperature=0.3)
        data = json.loads((msg.content or "{}").strip().strip("`").replace("json", "", 1))
        return data.get("topic", "")[:20], data.get("reason", "")[:60]
    except Exception:
        return (tags or "").split("/")[0][:20], "（归纳失败，退回首个标签）"


# --------------------------------------------------------------------- 调度

_trigger_lock = threading.Lock()
_next_auto = {"at": (int(time.time() // INTERVAL) + 1) * INTERVAL}
_pending = {"armed": False, "at": 0.0}
_surf_lock = threading.Lock()


def _tick(trigger: str, seed: str = "") -> None:
    with _surf_lock:                       # 手动和定时撞车时，后来的直接跳过
        try:
            run_surf(trigger, seed)
        except Exception as exc:
            q("INSERT INTO diary(ts,trigger,text,actions) VALUES(?,?,?,?)",
              (time.time(), "surf", f"（这一轮出门失败了：{type(exc).__name__} {exc}）", "[]"))


def request_user_tick(delay: float = 30.0) -> None:
    """收到反馈后延迟触发一轮，防止连点五下刷五轮。"""
    with _trigger_lock:
        _pending["armed"] = True
        _pending["at"] = time.time() + delay


def scheduler() -> None:
    threading.Thread(target=_tick, args=("auto",), daemon=True).start()   # 启动先来一轮
    while True:
        time.sleep(10)
        now = time.time()
        with _trigger_lock:
            armed, at = _pending["armed"], _pending["at"]
        if armed and now >= at:
            with _trigger_lock:
                _pending["armed"] = False
            _tick("user")
        elif now >= _next_auto["at"]:
            _next_auto["at"] = (int(now // INTERVAL) + 1) * INTERVAL
            threading.Thread(target=_tick, args=("auto",), daemon=True).start()


# --------------------------------------------------------------------- HTTP

app = FastAPI(title="阿浪 · B站游戏区冲浪")


class ChatIn(BaseModel):
    message: str


class FeedbackIn(BaseModel):
    bvid: str
    value: int


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/state")
def state():
    return {
        "interest": [dict(r) for r in q("SELECT topic,score,reason FROM interest ORDER BY score DESC")],
        "tokens_today": tokens_today(),
        "token_budget": TOKEN_BUDGET,
        "next_surf_in": max(0, int(_next_auto["at"] - time.time())),
        "last_surf": next((r["ts"] for r in q(
            "SELECT ts FROM diary WHERE trigger='surf' ORDER BY ts DESC LIMIT 1")), None),
        "read_limit": READ_LIMIT,
        "interval": INTERVAL,
    }


@app.get("/api/diary")
def diary():
    rows = q("SELECT ts,trigger,text,actions FROM diary ORDER BY ts DESC LIMIT 40")
    return [dict(r) for r in rows]


@app.get("/api/videos")
def videos():
    rows = q("SELECT bvid,title,author,tname,view,tags,url,created_at FROM videos "
             "ORDER BY created_at DESC LIMIT 60")
    return [dict(r) for r in rows]


@app.get("/api/chat")
def chat_history():
    return [dict(r) for r in q("SELECT role,content,ts FROM chat ORDER BY ts DESC LIMIT 40")][::-1]


@app.post("/api/chat")
def post_chat(body: ChatIn):
    reply = chat(body.message)
    # 用户说了话也要出门一趟（原话：「若用户有反馈或者对话，再进行刷十个视频」）。
    # 30 秒防抖会把连发几条消息合并成一轮，否则你会一次点出好几轮。
    request_user_tick()
    return {"reply": reply}


@app.post("/api/surf")
def manual_surf():
    threading.Thread(target=_tick, args=("manual",), daemon=True).start()
    return {"ok": True, "msg": "已经出门了，几秒后看日记。"}


@app.post("/api/feedback")
def feedback(body: FeedbackIn):
    topic, reason = derive_topic(body.bvid, body.value)
    q("INSERT INTO feedback(bvid,value,ts,topic,reason) VALUES(?,?,?,?,?)",
      (body.bvid, body.value, time.time(), topic, reason))
    if topic:
        # 用户反馈权重 0.7，比自己刷出来的观感更重
        _bump(topic, 0.18 * 0.7 * (1 if body.value > 0 else -1),
              f"用户{'好评' if body.value > 0 else '差评'}")
    request_user_tick()
    return {"topic": topic, "reason": reason}


@app.post("/api/ablation")
def ablation():
    """对照实验：忽略 feedback 表，在同样的状态下再跑一轮，把动作并排摆出来。"""
    baseline = q("SELECT actions FROM diary WHERE trigger='surf' ORDER BY ts DESC LIMIT 1")
    before = json.loads(baseline[0]["actions"]) if baseline else []
    after = run_surf("ablation", dry_run=True)
    return {"with_feedback": before, "without_feedback": after["actions"]}


if __name__ == "__main__":
    init_db()
    print(f"阿浪醒了。模型 {MODEL}｜每 {INTERVAL}s 出门一次｜每轮精读 {READ_LIMIT} 条")
    # 登录态先查，而且放在调度线程之前：它决定这轮能拿到多少语料，
    # 并发跑的话还会和第一轮冲浪抢同一个 session。
    try:
        who = bili.whoami()
        print(f"B 站登录态：{who['uname']}" if who["logged_in"] else
              "B 站未登录：评论只能拿 3 条预览、搜索更容易被限流。"
              "把 SESSDATA 填进 .env 的 BILI_SESSDATA 即可解开。")
    except BiliError as exc:
        print(f"B 站登录态没查成（不影响启动）：{exc}")
    threading.Thread(target=scheduler, daemon=True).start()
    print("打开 http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
