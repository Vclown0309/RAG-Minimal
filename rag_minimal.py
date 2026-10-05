"""RAG-Minimal：单文件、零第三方依赖的手写 RAG 全链路（教材版）。

一行读懂：建库 = 切块 + 向量化 + 落库；提问 = 向量化 → 余弦 top-k → 拼 Prompt → 生成。

对照《RAG 学习手册》章节阅读：
    chunk()    ↔ 2. 切块（标题结构切，超长窗口兜底）
    embed()    ↔ 3. 向量化（调 llama-server /v1/embeddings，纯 urllib 零依赖）
    store()    ↔ 4. SQLite 存储（BLOB 向量 + chunk_hash 幂等去重）
    search()   ↔ 5. 余弦相似度（手写点积/模长，无向量数据库）
    generate() ↔ 6. 拼 Prompt 开卷答题（答案可追溯：带参考源）

依赖：Python 3.10+（stdlib 仅用 sqlite3 / urllib / json）
前置：两个 llama-server 已启动（见 README 0.3 节）：
    embed: llama-server -m <嵌入模型> --embedding --pooling last -c 8192 --port 8081
    chat : llama-server -m <对话模型> -c 8192 --port 8080
    # 两个参数是性能保险，少一个都会卡：
    # -c 8192：限制上下文。不带它 llama.cpp 默认按模型训练上限（Qwen3 系 40960
    #   token）× 4 slots 建 KV cache → 单个 4B 服务吃 20GB+ 内存/显存，双服务打满。
    # -ngl 999：把模型层全放到显卡（有 NVIDIA 独显时）。新版 llama.cpp 默认已全卸载，
    #   显式写是为了兼容旧版本（旧版默认 -ngl 0 = 全 CPU，慢一个数量级）。
    # 另：Qwen3 系默认开"思考"，本代码已在 generate() 里关闭（enable_thinking=False）
    #   并限制输出长度（max_tokens=512）；换非 Qwen 模型若报错，删掉该键即可。

用法：
    python rag_minimal.py --build 文档.md   # 建库（可重复，幂等去重）
    python rag_minimal.py --ask "问题"       # 提问：检索 + 生成
    python rag_minimal.py --reset            # 清空知识库
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import urllib.request
from pathlib import Path

EMBED_URL = "http://127.0.0.1:8081/v1/embeddings"
CHAT_URL = "http://127.0.0.1:8080/v1/chat/completions"
DB = Path(__file__).with_name("kb.db")
WINDOW = 500  # 超长节内部窗口切的上限
TOP_K = 3


def _post(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read())


def embed(text: str) -> list[float]:
    """文字 → 向量。向量是一串数字，数字本身没意义，方向才表达语义。"""
    return _post(EMBED_URL, {"model": "x", "input": text})["data"][0]["embedding"]


def _is_head(s: str) -> bool:
    """标题判定：Markdown ATX（# 开头且去掉 # 后有内容）或手册原生（【】包裹）。"""
    if s.startswith("【") and s.endswith("】"):
        return True
    return s.startswith("#") and bool(s.lstrip("#").strip())


def chunk(text: str) -> list[str]:
    """标题结构切：一个标题一块。无标题文档退化为窗口切（兜底分支）。"""
    blocks: list[str] = []
    current: list[str] = []
    in_code = False
    head_lines: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.startswith("```"):  # 代码块开关：块内不做标题判定，避免注释行误判
            in_code = not in_code
        if not in_code and _is_head(s):
            if current:
                blocks.append("\n".join(current))
            current = [s]
            head_lines.append(s)
        elif s:
            current.append(s)
    if current:
        blocks.append("\n".join(current))

    if not head_lines:  # 无标题兜底：窗口切（500 字步长 450，重叠 50 防语义被切断）
        return [text[i : i + WINDOW] for i in range(0, len(text), WINDOW - 50)]
    # 超长节节内再窗口切（两级策略）
    out: list[str] = []
    for b in blocks:
        out += [b] if len(b) <= WINDOW else [b[i : i + WINDOW] for i in range(0, len(b), WINDOW - 50)]
    return out


def store(blocks: list[str]) -> None:
    """建表 + 存向量（BLOB）+ chunk_hash 幂等去重：同一文档重复上传不攒重复块。"""
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS chunks(id INTEGER PRIMARY KEY, hash TEXT UNIQUE, content TEXT, vector BLOB)")
    for b in blocks:
        h = hashlib.sha1(b.encode()).hexdigest()
        vec = embed(b)
        blob = json.dumps(vec).encode()  # 教材版用 JSON 存，可读性好；生产版换 struct 二进制
        con.execute("INSERT OR IGNORE INTO chunks(hash, content, vector) VALUES(?,?,?)", (h, b, blob))
    con.commit()
    print(f"入库 {len(blocks)} 块（去重后库内共 {con.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]} 块）")
    con.close()


def _cosine(v1: list[float], v2: list[float]) -> float:
    dot = sum(x * y for x, y in zip(v1, v2))
    n1 = sum(x * x for x in v1) ** 0.5
    n2 = sum(y * y for y in v2) ** 0.5
    return dot / (n1 * n2) if n1 and n2 else 0.0


def search(query: str, k: int = TOP_K) -> list[tuple[str, float]]:
    """问题向量化 → 和库里每块比余弦 → 取最像的 top-k。手写循环，没借向量数据库。"""
    qv = embed(query)
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT content, vector FROM chunks").fetchall()
    con.close()
    scored = sorted(
        ((content, _cosine(qv, json.loads(v))) for content, v in rows),
        key=lambda p: p[1], reverse=True,
    )
    return scored[:k]


def generate(query: str) -> None:
    """检索 → 拼 Prompt（根据以下资料回答）→ 生成 → 打印答案 + 参考源。"""
    hits = search(query)
    if not hits:
        print("知识库是空的：先 python rag_minimal.py --build 文档.md")
        return
    prompt = "\n".join(f"资料{i+1}：{c}" for i, (c, _) in enumerate(hits))
    prompt += f"\n问题：{query}\n请仅依据以上资料回答，不要编造资料外的内容。"
    # Qwen3/Qwen3.5 系默认开"思考"：不关掉会先输出几百上千 token 的英文推理，
    # 慢（CPU 上几十秒起步）且吃内存。max_tokens 限总输出长度，防模型无限生成。
    # 换非 Qwen 模型时若报 chat_template_kwargs 错，删掉该键即可。
    reply = _post(CHAT_URL, {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "chat_template_kwargs": {"enable_thinking": False},
    })
    print("答案：", reply["choices"][0]["message"]["content"])
    print("-" * 40)
    for i, (content, score) in enumerate(hits, 1):
        print(f"参考源{i}（相似度 {score:.3f}）：\n{content[:80]}...\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="RAG-Minimal 手写 RAG 全链路")
    ap.add_argument("--build", metavar="MD", help="把 Markdown 文档切块入库")
    ap.add_argument("--ask", metavar="问题", help="提问（检索 + 生成）")
    ap.add_argument("--reset", action="store_true", help="清空知识库")
    args = ap.parse_args()

    if args.reset:
        sqlite3.connect(DB).execute("DELETE FROM chunks").connection.commit()
        print("知识库已清空")
    elif args.build:
        store(chunk(Path(args.build).read_text(encoding="utf-8")))
    elif args.ask:
        generate(args.ask)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
