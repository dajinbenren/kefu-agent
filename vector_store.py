"""
vector_store.py —— 知识库：文档 → 切块 → 向量化 → 存入 Chroma → 可检索

职责一句话：把 data/ 里的 markdown 文档"腌"成能被语义搜索的一堆小块。
对外只有三个动作：
    build()   建库（切分 + 向量化 + 写入）
    search()  拿一句话，捞回最相关的几块原文
    stats()   体检：库到底正不正常（kb-site 最值钱的教训就沉淀在这里）

本版只做向量检索。BM25 混合召回等真实检索缺陷暴露后再升级——
先让系统跑起来，遇到具体失败案例再对症下药，这是工程节奏。
"""

import glob
import os

import chromadb

import config
import llm_client


# ---------- 1. 切分 ----------
def load_and_split() -> list[dict]:
    """读 data/*.md，切成小块。返回 [{text, source, idx}, ...]

    切分策略（kb-site 验证过的两段式）：
      1) 先按 ## 标题分节 —— 标题是天然的语义边界
      2) 超长的节再按滑动窗口细切 —— 窗口 400 字，每次只前进 400-80=320，
         多出的 80 字重叠，保证没有一句话被"切腰"
    """
    chunks: list[dict] = []

    for path in sorted(glob.glob(os.path.join(config.settings.DATA_DIR, "*.md"))):
        source = os.path.basename(path)
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()

        # --- 第一段：按标题分节 ---
        sections: list[tuple[str, str]] = []  # [(节标题, 节正文)]
        title, buf = lines[0].strip() if lines else "", []
        for line in lines[1:]:
            if line.startswith("#"):
                sections.append((title, "\n".join(buf).strip()))
                title, buf = line.strip(), []
            else:
                buf.append(line)
        sections.append((title, "\n".join(buf).strip()))

        # --- 第二段：长节滑动窗口 ---
        size, overlap = config.settings.CHUNK_SIZE, config.settings.CHUNK_OVERLAP
        step = size - overlap
        for sec_title, body in sections:
            if not body:
                continue
            if len(body) <= size:
                # 关键设计：标题要一起存进块里。
                # 单看"退款 5-7 个工作日"模型会懵：谁的政策？
                # 带上"# 星辰科技退换货政策"它才知道出处和主题。
                chunks.append({
                    "text": f"{sec_title}\n{body}",
                    "source": source,
                })
            else:
                for i in range(0, len(body), step):
                    piece = body[i:i + size]
                    chunks.append({
                        "text": f"{sec_title}\n{piece}",
                        "source": source,
                    })

    # 给每块编号（用于前端展示引用来源：退换货政策.md 第3块）
    for idx, c in enumerate(chunks):
        c["idx"] = idx
    return chunks


# ---------- 2. Chroma 句柄 ----------
def _collection():
    chroma = chromadb.PersistentClient(path=config.settings.CHROMA_DIR)
    return chroma.get_or_create_collection(name=config.settings.COLLECTION)


# ---------- 3. 建库 ----------
def build(force: bool = False) -> int:
    col = _collection()
    if col.count() > 0 and not force:
        print(f"[向量库] 已存在（{col.count()} 块），跳过。force=True 可强制重建。")
        return col.count()

    # 清旧数据用 Chroma 自己的 API，不删目录——
    # shutil.rmtree 在容器/只读环境会被拦（kb-site 真踩过）
    if col.count() > 0:
        chroma = chromadb.PersistentClient(path=config.settings.CHROMA_DIR)
        chroma.delete_collection(name=config.settings.COLLECTION)
        col = _collection()

    chunks = load_and_split()
    print(f"[切分] {len(chunks)} 块，来自 {len(set(c['source'] for c in chunks))} 个文件")
    vectors = llm_client.embed([c["text"] for c in chunks])

    col.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c["source"], "idx": c["idx"]} for c in chunks],
    )
    print(f"[建库] 完成：{len(chunks)} 块 → {config.settings.CHROMA_DIR}")
    return len(chunks)


# ---------- 4. 检索 ----------
def search(query: str, top_k: int | None = None) -> list[dict]:
    """把用户问题也变成向量，去库里找方向最接近的几块。"""
    col = _collection()
    if top_k is None:
        top_k = config.settings.TOP_K
    if col.count() == 0:
        return []

    q_vec = llm_client.embed([query])[0]
    n = min(top_k, col.count())  # 防御：要的比库里多，就全给
    res = col.query(query_embeddings=[q_vec], n_results=n)

    return [
        {
            "text": doc,
            "source": meta["source"],
            "idx": meta["idx"],
            "rank": i + 1,
        }
        for i, (doc, meta) in enumerate(zip(res["documents"][0], res["metadatas"][0]))
    ]


# ---------- 5. 体检 ----------
def stats() -> dict:
    """kb-site 传下来的最重要教训：
    上手任何 RAG，第一件事打印 chunk 数——
    如果 库里块数 <= TOP_K，说明检索框比候选池还大，
    每次都是全量注入，测的是一个假检索系统。"""
    col = _collection()
    total = col.count()
    sources = {}
    if total:
        for m in col.get(include=["metadatas"])["metadatas"]:
            sources[m["source"]] = sources.get(m["source"], 0) + 1
    return {
        "total": total,
        "sources": sources,
        "top_k": config.settings.TOP_K,
        "degenerate": 0 < total <= config.settings.TOP_K,  # True = 检索已退化
    }


if __name__ == "__main__":
    import sys

    force = "--rebuild" in sys.argv
    build(force=force)
    print()
    s = stats()
    print(f"[体检] 总块数={s['total']}  TOP_K={s['top_k']}")
    for src, n in s["sources"].items():
        print(f"       {src}: {n} 块")
    print(f"[体检] 检索退化(degenerate) = {s['degenerate']}  {'⚠️ 库比TOP_K还小，全量注入!' if s['degenerate'] else 'OK，检索在真实工作'}")

    # 冒烟测试：三个典型客户问法，看召回对不对
    print("\n[冒烟] 三个典型客户问题的召回：")
    for q in ["你们是不是诈骗团伙", "老人家不会用手机怎么退货", "多久能到"]:
        hits = search(q)
        print(f"  Q: {q}")
        for h in hits:
            print(f"    [{h['rank']}] {h['source']} 第{h['idx']}块 | {h['text'][:40]}...".replace("\n", " "))
