# -*- coding: utf-8 -*-
"""
crawler.py —— B 站游戏区只读采集层

只负责把数据拿回来并整理成统一结构：不碰数据库、不碰 LLM、不碰业务判断。

设计要点（均经真实请求验证）：
  * 必须先用首页拿服务器签发的 cookie（buvid3 / b_nut），再调指纹接口补 buvid4，
    缺了会被风控返回 -412。
  * 搜索和评论要 WBI 签名（2023-03 起），密钥从 nav 接口取、约 12 小时有效，这里缓存。
  * 排行榜不需要签名，是最稳的兜底路径。
  * SESSDATA（可选，见 _auth）是解开评论翻页的那把钥匙。

实测的登录差别：未登录也能刷榜（96 条）、能搜（20 条）、能拿弹幕（几百条），
只有评论被卡死——服务端只给 3 条预览，ps 给多大都没用，cursor.is_end 直接为 True。
评论是这里唯一有实际内容的长文本，所以未登录时 RAG 的语料薄得像纸。

礼貌采集：请求间隔 >= 1 秒并带抖动，失败指数退避，单次运行在百次以内。
"""
from __future__ import annotations

import html
import hashlib
import json
import os
import random
import re
import time
from urllib.parse import urlencode

import requests

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

API = "https://api.bilibili.com"

# B 站游戏区及其子分区，供 agent 自己挑去哪刷
GAME_RIDS = {
    4: "游戏区（总）",
    17: "单机游戏",
    171: "电子竞技",
    172: "手机游戏",
    65: "网络游戏",
    173: "桌游棋牌",
    136: "音游",
}

# 分区名映射，按 tid 查。tid 与 rid 是同一套 id 空间，所以直接在 GAME_RIDS 上派生，
# 不另写一张表——手抄的映射迟早和后面对不上。
# 存在的理由：view 接口现在 tname / tname_v2 都是空串，只剩 tid；照老字段读会拿到
# 空串，而空串会被 backend 原样写进向量库当成「分区 」——那是字段没了，不是没抓到。
# 查不到的一律渲染成「未知分区(tid=N)」，绝不留空：空串是「没抓到」，猜错的名字是
# 「错误知识」，后者更糟，因为它会被当成一条规律记住。
TID_NAMES = dict(GAME_RIDS)

# 「最新投稿」入口接受的 rid。实测只有总区有数据，细分子分区一律空——和 ranking/v2
# 一样只认总区。不开放 rid=0：那是全站，会混进游戏区之外的内容。
NEWLIST_RIDS = (4,)

# 搜索排序。接口接受这几种，实测值得用的只有 pubdate：默认的 totalrank 是相关度序，
# 同一个词每次返回同一批人、同一批视频，连着搜等于把同一批结果看七遍（真库里发生过）。
# pubdate 与它几乎不重叠（同一词只重叠 1/20），是打破这种自锁最省的一步。
# click 按播放排，等于往爆款池推；scores 常常和 totalrank 没区别。
SEARCH_ORDERS = ("totalrank", "pubdate", "click", "scores")

# WBI 混入表的固定重排顺序，B 站前端硬编码，不是可调参数
_MIXIN_TAB = [46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35,
              27, 43, 5, 49, 33, 9, 42, 19, 29, 28, 14, 39, 12, 38, 41, 13,
              37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 60, 51, 30, 4,
              22, 25, 54, 21, 56, 59, 6, 63, 57, 62, 11, 36, 20, 34, 44, 52]

_TAG_RE = re.compile(r"<[^>]+>")
_DM_RE = re.compile(r'<d p="[^"]*">([^<]*)</d>')


def _clean(text: str) -> str:
    """搜索结果标题里带 <em class='keyword'> 高亮标签，要去掉并反转义。"""
    return html.unescape(_TAG_RE.sub("", text or "")).strip()


class BiliError(RuntimeError):
    """接口返回非 0 code，或重试后仍被风控拦截。

    code 单独留着：调用方偶尔要按 code 定退路（见 browse_region——-400 表示「这个 rid
    不被该接口接受」，不是真失败，可以换个接口重来）。网络失败、重试耗尽这类没有服务端
    code 的情况，code 为 None。
    """

    def __init__(self, msg: str, code: int | None = None):
        super().__init__(msg)
        self.code = code


class BiliThrottled(BiliError):
    """搜索被软限流挡下。接口 code=0、不报错，只是整个响应只剩一个 v_voucher。

    单独一个类型，是因为它跟「这个词真没内容」长得一模一样（都是空列表），
    但含义完全相反。混在一起返回 [] 的话，调用方会把限流读成搜索结果，
    把它当成一条关于世界的知识记下来——真库里已经这么发生过一次了。
    """


class BiliClient:
    def __init__(self, min_interval: float = 1.0, timeout: int = 15, max_retry: int = 3):
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retry = max_retry
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": UA,
            "Referer": "https://www.bilibili.com/",
            "Accept": "application/json, text/plain, */*",
            "Origin": "https://www.bilibili.com",
        })
        self._wbi: str | None = None
        self._wbi_at: float = 0.0
        self._last_call: float = 0.0
        self._warm()

    # ------------------------------------------------------------------ 基础设施

    def _throttle(self) -> None:
        """两次请求之间至少隔 min_interval 秒，再加 0~0.6 秒抖动，避免节奏过于机器化。"""
        gap = time.time() - self._last_call
        wait = self.min_interval - gap
        if wait > 0:
            time.sleep(wait)
        time.sleep(random.uniform(0, 0.6))
        self._last_call = time.time()

    def _auth(self) -> None:
        """挂登录 cookie。这是花最小代价解开评论限制的那一步。

        未登录时评论只返回 3 条预览，那 ps 给多大都没用——是服务端的授权门槛，不是限流。
        SESSDATA 等价于账号登录态，只从环境变量（.env）读，不落进代码库。
        没配就静默跳过（仍能刷榜/搜/拿弹幕，只是语料薄）；重新预热会再挂一次，
        顺带覆盖掉过期的那份。
        """
        raw = os.environ.get("BILI_COOKIE", "").strip()
        if raw.lower().startswith("cookie:"):
            raw = raw.split(":", 1)[1]           # 从 devtools 整行复制过来的情况
        for part in raw.split(";"):
            if "=" in part:
                k, v = part.split("=", 1)
                if k.strip():
                    self.s.cookies.set(k.strip(), v.strip(), domain=".bilibili.com")
        sess = os.environ.get("BILI_SESSDATA", "").strip()
        if sess:
            self.s.cookies.set("SESSDATA", sess, domain=".bilibili.com")

    def whoami(self) -> dict:
        """当前身份：确认 SESSDATA 挂上没有、过没过期。

        从别处看不出来——cookie 过期后接口不报错，只是评论悄悄退回 3 条，日志毫无异常。
        """
        self._throttle()
        try:
            nav = self.s.get(f"{API}/x/web-interface/nav", timeout=self.timeout).json()
        except Exception as exc:
            raise BiliError(f"nav 请求失败：{exc}") from exc
        d = nav.get("data") or {}
        return {"logged_in": bool(d.get("isLogin")), "uname": d.get("uname") or "",
                "mid": d.get("mid") or 0}

    def _warm(self) -> None:
        """访问首页让服务端签发 cookie，再用指纹接口补 buvid3 / buvid4。"""
        self._auth()          # 放在 try 外面：挂 cookie 不该被任何预热失败吞掉
        try:
            self.s.get("https://www.bilibili.com/", timeout=self.timeout)
            spi = self.s.get(f"{API}/x/frontend/finger/spi", timeout=self.timeout).json()
            data = spi.get("data") or {}
            for name, value in (("buvid3", data.get("b_3")), ("buvid4", data.get("b_4"))):
                if value:
                    self.s.cookies.set(name, value, domain=".bilibili.com")
            self.s.cookies.set("b_nut", str(int(time.time())), domain=".bilibili.com")
        except Exception:
            # 拿不到指纹不致命：cookie 预热失败的请求会在 _get 里触发一次重新预热
            pass

    def _mixin_key(self) -> str:
        """从 nav 接口取 img_key/sub_key 拼出 mixin_key，缓存 12 小时。"""
        if self._wbi and time.time() - self._wbi_at < 12 * 3600:
            return self._wbi
        nav = self.s.get(f"{API}/x/web-interface/nav", timeout=self.timeout).json()
        wbi = ((nav.get("data") or {}).get("wbi_img") or {})
        img = wbi.get("img_url", "").rsplit("/", 1)[-1].split(".")[0]
        sub = wbi.get("sub_url", "").rsplit("/", 1)[-1].split(".")[0]
        if not img or not sub:
            raise BiliError("取不到 WBI 密钥，nav 接口返回异常")
        raw = img + sub
        self._wbi = "".join(raw[i] for i in _MIXIN_TAB)[:32]
        self._wbi_at = time.time()
        return self._wbi

    def _sign(self, params: dict) -> dict:
        """WBI 签名：按 key 字典序编码，拼上 wts 和 mixin_key 取 md5。"""
        p = dict(params)
        p["wts"] = int(time.time())
        query = urlencode(sorted(p.items()))
        p["w_rid"] = hashlib.md5((query + self._mixin_key()).encode()).hexdigest()
        return p

    def _get(self, path: str, params: dict | None = None, signed: bool = False) -> dict:
        """统一出口：限速 -> 签名 -> 请求 -> 解包 -> 风控/网络错误重试。"""
        for attempt in range(self.max_retry + 1):
            self._throttle()
            try:
                q = self._sign(params or {}) if signed else (params or {})
                payload = self.s.get(API + path, params=q, timeout=self.timeout).json()
            except Exception as exc:
                if attempt == self.max_retry:
                    raise BiliError(f"{path} 请求失败：{exc}") from exc
                time.sleep(1.5 * (attempt + 1))
                continue

            code = payload.get("code")
            if code == 0:
                return payload.get("data") or {}
            # -412/-352/-401 是风控或密钥过期：重新预热 + 刷新密钥，退避后重试。
            # 这里没有 -403：那是评论接口的「请求太密」，实测连发几次后整条接口都会
            # 这么回且几分钟不退。重试会再 _warm()，每退一次反而多打 2 个请求，
            # 越撞越死——碰上它要立刻停手，不是重试。
            if code in (-412, -352, -401) and attempt < self.max_retry:
                self._warm()
                self._wbi = None
                time.sleep(2.0 * (attempt + 1))
                continue
            raise BiliError(f"{path} 返回 code={code} msg={payload.get('message')}", code)

    # ------------------------------------------------------------------ 对外接口

    @staticmethod
    def _throttled(data: dict) -> bool:
        """软限流的判据：没有 result，且 numResults 这个键整个不见了。

        看键在不在，不看值真假——真搜到 0 条时 numResults 是 0、值也是假，用取值判断
        会把「真没内容」误判成「被挡了」，正好反过来。被挡的响应里只剩一个 v_voucher。
        """
        return not data.get("result") and "numResults" not in data

    def search(self, keyword: str, page: int = 1, order: str = "totalrank") -> list[dict]:
        """关键词搜索视频，返回候选列表（只有元信息，供 agent 挑要不要精读）。

        order 见 SEARCH_ORDERS。默认 totalrank 是相关度序，同一个词每次返回同一批人、
        同一批视频，连着搜等于把同一批结果看七遍；换 pubdate 能翻出完全不同的一批
        （实测只重叠 1/20）——但它只解决「同一批人」，不解决「同一个话题」，主题还得换词。

        连续搜索会被软性限流：接口返回 code=0 但 result 为空，不报错。实测冷却 10~15 秒，
        所以两次重试必须隔得比这个窗口长——原来的 4 秒、9 秒等于在同一个冷却里连撞三次。

        撞完仍被挡就抛 BiliThrottled：调用方必须知道这是「被挡了」而不是「这个词没内容」，
        否则会把限流读成一条关于世界的知识记进日记。
        """
        for wait in (0.0, 6.0, 16.0):
            if wait:
                time.sleep(wait)
            data = self._get("/x/web-interface/wbi/search/type",
                             {"search_type": "video", "keyword": keyword,
                              "page": page, "order": order},
                             signed=True)
            if self._throttled(data):
                continue
            # 搜索结果的字段名和榜单不同（作者在顶层的 author、播放叫 play），
            # 但形状必须和别的路径一致，所以照样过 _pack。
            return [x for x in (self._pack(it.get("bvid"), it.get("title"), it.get("author"),
                                          it.get("duration"), it.get("play"), it.get("pubdate"))
                                for it in data.get("result") or [] if it.get("bvid"))]
        raise BiliThrottled(keyword)

    @staticmethod
    def _pack(bvid, title, author, duration, play, pubdate) -> dict:
        """候选条目的统一形状，四条路径（搜索 / 热榜 / 子分区 / 最新）共用。

        形状必须一致：backend 是硬取键的，少一个键就是 KeyError，会被工具循环
        的兜底 except 吞成「工具执行异常」——那又是一次「没抓到」被说成别的。

        play 单独判 None 而不是写 `play or ""`：最新投稿的播放量真的是 0，
        折叠成空串就把「刚发布、还没人看」这条信息抹掉了。
        """
        return {"bvid": bvid, "title": _clean(title or ""), "author": author,
                "duration": duration, "pubdate": pubdate or 0,
                "play_text": "" if play is None else str(play)}

    @classmethod
    def _pack_rows(cls, items: list) -> list[dict]:
        """ranking/v2 与 newlist 的条目字段名相同，统一在这里收口成 _pack 的形状。"""
        return [x for x in (cls._pack(it.get("bvid"), it.get("title"),
                                      (it.get("owner") or {}).get("name"),
                                      it.get("duration"), (it.get("stat") or {}).get("view"),
                                      it.get("pubdate")) for it in items) if x["bvid"]]

    def browse_region(self, rid: int = 4, mode: str = "hot") -> list[dict]:
        """刷一个分区的候选列表。

        mode="hot" 走热度榜：不需要签名，最稳，但一天之内基本不动。rid=4（游戏区总）、
        rid=0（全站）走 ranking/v2 一次约百条；细分分区（17/171…）已不在它接受的列表里，
        一律返回 -400——那不是失败，是「接口不认识这个 rid」，于是退回旧接口
        ranking/region，它仍按 sub-rid 工作，代价是只给榜单前十名左右。

        mode="new" 走最新投稿：时间序、每天翻新，见 newlist()。

        子分区候选天生比总区少一个量级，是接口现状不是这里偷懒，广撒网还是得靠总区榜或搜索。
        """
        if mode == "new":
            return self.newlist(rid)
        try:
            data = self._get("/x/web-interface/ranking/v2", {"rid": rid, "type": "all"})
        except BiliError as exc:
            if exc.code != -400:
                raise
            return self._browse_sub_region(rid)
        return self._pack_rows(data.get("list") or [])

    def newlist(self, rid: int = 4, page: int = 1) -> list[dict]:
        """分区最新投稿，时间序。这是让候选池真的翻新的那个入口。实测：
          * 只有 rid=4（游戏区总）有数据；细分子分区一律返回空 archives，和 ranking/v2
            一样只认总区。
          * 每页 50 条，翻到 pn≈10（约 500 条）见底，pn=30 与 pn=10 是同一批。
          * 与热度榜几乎不重叠，且每天自己是新的。

        只开放 rid=4，不开放 rid=0：全站会混进汽车保养这类无关内容，而刷回来的每一条
        都会进记忆、进日记，最后变成一条关于「游戏区」的错误知识。真要开，工具文案必须
        先讲清楚那是全站。
        """
        data = self._get("/x/web-interface/newlist", {"rid": rid, "ps": 50, "pn": page})
        if "archives" not in data:
            # 照 _throttled 的思路：看键在不在，不看值真假。
            # 缺键是「没拿到」，和「这个分区没有新投稿」长得一模一样但含义相反。
            raise BiliError(f"newlist 没返回 archives 键（rid={rid}），"
                            f"这是没抓到，不代表该分区没有新投稿")
        return self._pack_rows(data.get("archives") or [])

    def _browse_sub_region(self, rid: int) -> list[dict]:
        """细分分区的退路：合并三日榜/周榜/日榜，去重后约二三十条。

        三个榜的头部高度重叠，三次请求换不来三倍候选，增量主要在中后段；但仍值——
        单次只给十条，对「这个分区最近在聊什么」来说太薄了。

        某个榜挂了不拖垮整次浏览；全都挂了必须抛出去——不能把失败读成「这个分区
        没内容」，那会变成一条错误知识被记下来。
        """
        out: list[dict] = []
        seen: set = set()
        last: BiliError | None = None
        for day in (3, 7, 1):
            try:
                rows = self._get("/x/web-interface/ranking/region",
                                 {"rid": rid, "day": day})
            except BiliError as exc:
                last = exc
                continue
            for it in (rows if isinstance(rows, list) else []):
                bvid = it.get("bvid")
                if not bvid or bvid in seen:
                    continue
                seen.add(bvid)
                # 这个接口的字段名和另外两条路径都不一样：作者在 author（不是
                # owner.name）、播放叫 play、发布时间叫 create。_pack 统一形状。
                out.append(self._pack(bvid, it.get("title"), it.get("author"),
                                      it.get("duration"), it.get("play"),
                                      it.get("create")))
        if not out and last:
            raise last
        return out

    def video_detail(self, bvid: str) -> dict:
        d = self._get("/x/web-interface/view", {"bvid": bvid})
        stat = d.get("stat") or {}
        tid = d.get("tid")
        return {
            "bvid": d.get("bvid"),
            "aid": d.get("aid"),
            "cid": d.get("cid"),
            "title": d.get("title"),
            "desc": d.get("desc") or "",
            # 接口哪天又开始给名字了就优先用它的；现状是只给 tid，那就查 TID_NAMES；
            # 两样都没有也不能交空串出去。
            "tname": (d.get("tname") or TID_NAMES.get(tid)
                      or (f"未知分区(tid={tid})" if tid else "")),
            "tid": tid,                          # 只在自检/审计里用，入库时不存
            "author": (d.get("owner") or {}).get("name"),
            "mid": (d.get("owner") or {}).get("mid"),
            "view": stat.get("view", 0),
            "like": stat.get("like", 0),
            "reply": stat.get("reply", 0),
            "danmaku_count": stat.get("danmaku", 0),   # 注意：这是数量。弹幕正文在 read_video 里另放 danmaku
            "pubdate": d.get("pubdate"),
            "duration": d.get("duration"),
            "url": f"https://www.bilibili.com/video/{bvid}",
        }

    def video_tags(self, bvid: str) -> list[str]:
        data = self._get("/x/tag/archive/tags", {"bvid": bvid})
        return [t.get("tag_name") for t in (data or []) if t.get("tag_name")]

    def comments(self, aid: int, ps: int = 20, mode: int = 2, limit: int = 20) -> list[dict]:
        """取评论。mode=2 按时间（数量足），mode=3 按热度。

        评论是这里唯一有实际内容的长文本：登录后一页 20 条，未登录只有 3 条且
        cursor.is_end 直接为 True，循环第一轮就退出——不是这里没写翻页，是服务端不让翻。

        ⚠️ 翻页（limit > ps）目前是坏的，别开：第一页走 pn 正常，第二页改用 pagination_str
        传 cursor.next 后接口 code=0 不报错，却每次把第一页原样再发一遍，去重后一条新的
        都没有（紧凑 JSON 也试过，不是空格编码问题）。所以默认 limit=20 只取一页。

        循环里的 break 是安全网：拿不到新东西就停，不会空转，也不会把「翻不动」误当成
        「这个视频没评论」——那会是一条错误知识入库。
        """
        out: list[dict] = []
        seen: set = set()
        cursor_next: str | None = None
        for _ in range(max(1, -(-limit // ps))):          # 向上取整
            params: dict = {"oid": aid, "type": 1, "mode": mode, "ps": ps}
            if cursor_next is None:
                params["pn"] = 1
            else:
                # 必须用紧凑 JSON：默认的 json.dumps 会在冒号后加空格，
                # 那个空格经 urlencode 变成 '+'，服务端解析不出 offset，
                # 于是把第一页原样再发一遍——看起来「翻页成功」其实一条没动。
                params["pagination_str"] = json.dumps({"offset": cursor_next},
                                                      separators=(",", ":"))
            try:
                data = self._get("/x/v2/reply/wbi/main", params, signed=True)
            except BiliError:
                if out:
                    break        # 已经拿到一部分，别为了下一页把整次精读搞崩
                raise
            added = 0
            for c in data.get("replies") or []:
                msg = ((c.get("content") or {}).get("message") or "").strip()
                if not msg or c.get("rpid") in seen:
                    continue
                seen.add(c.get("rpid"))
                out.append({
                    "rpid": c.get("rpid"),
                    "message": msg,
                    "like": c.get("like", 0),
                    "uname": (c.get("member") or {}).get("uname"),
                    "ctime": c.get("ctime"),
                })
                added += 1
            cursor = data.get("cursor") or {}
            cursor_next = str(cursor.get("next") or "")
            # is_end 是未登录时的常态；没有新增也说明翻不动了，两种都收工
            if cursor.get("is_end") or not added or not cursor_next:
                break
        return out[:limit]

    def danmaku(self, cid: int) -> list[str]:
        """取弹幕（XML，免签名，单次约 300~400 条）。

        存在的理由：未登录时评论只有 3 条，弹幕是唯一够量的观众反应文本，没有它
        RAG 的语料薄得像纸。填了 SESSDATA 后评论能翻到几十条，弹幕就从「唯一」退成
        「补充」——但它免签名、不占 WBI 额度，仍是最便宜的那份语料。
        """
        self._throttle()
        try:
            resp = self.s.get(f"https://comment.bilibili.com/{cid}.xml", timeout=self.timeout)
            resp.encoding = "utf-8"          # 服务端 charset 头不准，不显式指定会乱码
            return [html.unescape(t).strip() for t in _DM_RE.findall(resp.text) if t.strip()]
        except Exception as exc:
            raise BiliError(f"弹幕请求失败：{exc}") from exc

    def read_video(self, bvid: str, comment_size: int = 20,
                   comment_limit: int = 20) -> dict:
        """精读一个视频：详情 + 标签 + 评论 + 弹幕一次拿全，这是入库的基本单位。

        评论默认停在第一页（登录后 20 条，未登录永远 3 条）：登录这一下就把语料从
        3 条变成 20 条，是本项目性价比最高的一步。翻页逻辑在 comments() 里写好了但
        尚未验证通过（第二页总是第一页的原样重复），所以别把 limit 调大——现在调大
        只是每条视频白打 1~2 个请求。
        """
        detail = self.video_detail(bvid)
        detail["tags"] = self.video_tags(bvid)
        detail["comments_error"] = ""
        try:
            detail["comments"] = self.comments(detail["aid"], ps=comment_size,
                                               limit=comment_limit)
        except BiliError as exc:
            # 评论接口被限流时会持续好几分钟（实测 -403 不退），不该把详情、
            # 标签、弹幕一起赔进去——那等于整轮精读全废，12 条全挂。
            # 降级继续，但必须留记号：调用方要能说出「这次没抓到」，
            # 否则空评论会被读成「这视频没人评论」，那是错误的知识。
            detail["comments"] = []
            detail["comments_error"] = str(exc)
        detail["danmaku"] = self.danmaku(detail["cid"])
        return detail

    def close(self) -> None:
        self.s.close()


# ---------------------------------------------------------------------- 自检

if __name__ == "__main__":
    import io
    import sys
    from pathlib import Path

    import envfile

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    # 直接跑本文件时没人替我们读 .env（那一直是 backend 的活），但自检第一件事
    # 就是查登录态——不读的话，明明填了 cookie 也会报「未登录」，把最该相信的
    # 那行诊断变成假消息。
    envfile.load(Path(__file__).resolve().parent / ".env")

    t0 = time.time()
    cli = BiliClient()

    print("[0] 登录态")
    who = cli.whoami()
    if who["logged_in"]:
        print(f"    已登录：{who['uname']}（mid={who['mid']}），评论可以翻页")
    else:
        print("    未登录。评论只会拿到 3 条预览——把 SESSDATA 填进 .env 的")
        print("    BILI_SESSDATA 即可解开。刷榜/搜索/弹幕不受影响。")

    print("[1] 排行榜（游戏区总 rid=4）")
    hot = cli.browse_region(4)
    print(f"    {len(hot)} 条，例如：{hot[0]['title'][:40]} | {hot[0]['author']}")

    print("[2] 搜索 '黑神话悟空'")
    hits = cli.search("黑神话悟空")
    print(f"    {len(hits)} 条，例如：{hits[0]['title'][:40]}")

    print("[3] 精读第一条")
    v = cli.read_video(hits[0]["bvid"])
    print(f"    《{v['title'][:36]}》 by {v['author']}")
    print(f"    播放 {v['view']} / 评论 {v['reply']} / 标签 {v['tags'][:5]}")
    # 这行是 tname 修复的验收：打出分区名说明映射命中，打出「未知分区(tid=N)」
    # 说明 TID_NAMES 还没覆盖到它——那也不是抓取失败，是映射表该补了。
    print(f"    分区 {v['tname'] or '（空）'}（tid={v['tid']}）")
    n = len(v["comments"])
    if v.get("comments_error"):
        print(f"    ⚠️ 评论没抓到：{v['comments_error']}")
        print("       （评论接口被限流了，隔几分钟再试；详情/标签/弹幕不受影响）")
    else:
        hint = "" if who["logged_in"] or n > 3 else "（未登录的上限就是 3 条）"
        print(f"    抓到评论 {n} 条{hint}，"
              f"首条：{v['comments'][0]['message'][:40] if n else '（无）'}")
    print(f"    抓到弹幕 {len(v['danmaku'])} 条，样例：{v['danmaku'][:3]}")

    print("[4] 最新投稿（rid=4）与热度榜是不是两份池子")
    fresh = cli.newlist(4)
    same = len({h["bvid"] for h in fresh} & {h["bvid"] for h in hot})
    print(f"    {len(fresh)} 条，与上面热榜重叠 {same} 条——重叠少就说明 "
          f"mode=new 确实换了一批人，不是同一批换个顺序")
    if fresh:
        print(f"    例如：{fresh[0]['title'][:40]} | {fresh[0]['author']}")

    print(f"\n全部通过，耗时 {time.time() - t0:.1f}s")
