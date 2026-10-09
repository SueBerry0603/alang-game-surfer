# -*- coding: utf-8 -*-
"""
envfile.py —— 读取项目根目录下的 .env

单独一个文件，是因为有两个入口要用它：
  * backend.py 启动时
  * crawler.py 的自检（直接跑 `python crawler.py` 时没人替它加载配置，
    而自检第一件事就是查登录态）
两边各写一份的话迟早会不一致，而这里的不一致会表现为「cookie 明明填了，
自检却说未登录」——正是最难查的那种。

手写解析，省掉 python-dotenv 这个依赖。
"""
from __future__ import annotations

import os
from pathlib import Path


def load(path: Path) -> None:
    """把 .env 里的键值塞进 os.environ。

    用 setdefault 而不是覆盖：真实环境变量优先于文件，命令行临时覆盖
    （`BILI_SESSDATA=xxx python backend.py`）才有意义。

    注意由此推出的一个坑：同名键以**先出现的**为准。保留原文件里已经
    存在的空 `KEY=` 行、再在末尾追加一行真值，是不生效的。
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())
