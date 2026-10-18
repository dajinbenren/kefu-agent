#!/usr/bin/env python3
# eval_bench.py —— kefu-agent 评测基准：把"能跑"变成"可比较的证据"
"""
为什么存在这个文件：
    简历写"抗幻觉""省 token"都是空话，面试官一句"怎么测的、数字多少"就破功。
    这个脚本用一组标注好的客户问题，同时跑 RAG 和 Agent 两个内核，
    自动算出能直接填进简历的硬指标：
        - 首字延迟 TTFT（含检索 / 工具调用成本，对比两模式 trade-off）
        - Recall@5（检索质量：期望的知识库块有没有进前 5）
        - 来源命中率（该模式返回的 sources 是否含正确答案出处）
        - 幻觉率（答案是否出现语料里根本没有的编造事实，如把 15 天说成 7 天）
        - 漏答率（关键事实没答出来 = 假阴性）
        - Agent 工具调用合规率（政策题必调 search_kb、订单题必调 query_order、闲聊零工具）

用法：
    D:/DailySoftWare/Python/python.exe eval_bench.py                # 全量跑 RAG+Agent
    D:/DailySoftWare/Python/python.exe eval_bench.py --modes rag    # 只跑 RAG
    D:/DailySoftWare/Python/python.exe eval_bench.py --limit 5      # 先冒烟 5 条
    D:/DailySoftWare/Python/python.exe eval_bench.py --category 退货政策
    D:/DailySoftWare/Python/python.exe eval_bench.py --judge        # 额外开 LLM 裁判（多花 token）

前提：.env 里要有 DASHSCOPE_API_KEY；依赖同项目 requirements.txt。
产物：控制台汇总 + eval_report.json（运行产物，已加进 .gitignore，不进仓库）。
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)  # 保证 data/ .env chroma_db 相对路径正确

import config
import vector_store
import kernel_rag
import kernel_agent

# ============================================================
# 评测集（金标准来自 data/*.md 真实语料）
#   category        分类
#   q               客户问题
#   expected_source 答案应来自哪篇语料（用于 Recall@5 与来源命中；订单/超纲题留空）
#   must_include    答案应包含的关键事实（任一匹配即算命中；匹配时忽略空格）
#   must_not        幻觉标志（答案里出现任何一个 = 编造事实）
#   expect_unknown  超纲题=True，期望答案含"未提及/联系人工"等认输表述
#   expect_tool     Agent 期望调用的工具：search_kb / query_order / None(不该调)
#   note            考察点（写进报告）
# ============================================================
BENCH = [
    # ---------- 退货政策（核心：7天 vs 15天 对照）----------
    {"category": "退货政策", "q": "你们家七天无理由退货怎么操作？",
     "expected_source": "退换货政策.md", "must_include": ["15 天"],
     "must_not": ["7天", "七天", "7 天"], "expect_tool": "search_kb",
     "note": "幻觉对照题：语料写 15 天，模型若凭常识答 7 天即幻觉，检验 RAG-as-Tool 是否焊死在资料上"},
    {"category": "退货政策", "q": "质量问题退货有时间限制吗？",
     "expected_source": "退换货政策.md", "must_include": ["30 天"], "must_not": [],
     "expect_tool": "search_kb", "note": "质量问题 30 天，验证检索到正确条款"},
    {"category": "退货政策", "q": "无理由退货的运费谁出？",
     "expected_source": "退换货政策.md", "must_include": ["客户承担"], "must_not": [],
     "expect_tool": "search_kb", "note": "无理由退货寄回运费客户承担"},
    {"category": "退货政策", "q": "退款一般多久到账？",
     "expected_source": "退换货政策.md", "must_include": ["5-7", "5到7"], "must_not": [],
     "expect_tool": "search_kb", "note": "退款 5-7 个工作日原路返回"},
    {"category": "退货政策", "q": "哪些商品不能无理由退货？",
     "expected_source": "退换货政策.md", "must_include": ["定制", "贴身衣物"], "must_not": [],
     "expect_tool": "search_kb", "note": "不支持无理由的品类"},
    {"category": "退货政策", "q": "我要申请退货，入口在哪？",
     "expected_source": "退换货政策.md", "must_include": ["我的-订单", "申请售后"], "must_not": [],
     "expect_tool": "search_kb", "note": "退货入口路径"},
    {"category": "退货政策", "q": "质量问题的运费怎么算？",
     "expected_source": "退换货政策.md", "must_include": ["星辰科技承担", "免费上门"], "must_not": [],
     "expect_tool": "search_kb", "note": "质量问题退换货星辰承担 + 免费上门取件"},
    {"category": "退货政策", "q": "退换货政策你帮我讲讲吧",
     "expected_source": "退换货政策.md", "must_include": ["15 天"], "must_not": ["7天", "七天"],
     "expect_tool": "search_kb", "note": "泛问政策，验证是否主动检索而非凭记忆"},

    # ---------- 配送说明 ----------
    {"category": "配送", "q": "一线城市多久能收到货？",
     "expected_source": "配送说明.md", "must_include": ["24 小时"], "must_not": ["3天", "三天"],
     "expect_tool": "search_kb", "note": "一线/省会发货后 24 小时内；防模型答 3 天"},
    {"category": "配送", "q": "满多少包邮？",
     "expected_source": "配送说明.md", "must_include": ["99"], "must_not": [],
     "expect_tool": "search_kb", "note": "实付满 99 包邮"},
    {"category": "配送", "q": "偏远地区运费怎么算？",
     "expected_source": "配送说明.md", "must_include": ["15 元", "15元"], "must_not": [],
     "expect_tool": "search_kb", "note": "偏远加收 15 元远程运费"},
    {"category": "配送", "q": "物流好几天没更新怎么办？",
     "expected_source": "配送说明.md", "must_include": ["异常件", "24 小时"], "must_not": [],
     "expect_tool": "search_kb", "note": "超 72h 未更新 = 异常件，24h 反馈"},
    {"category": "配送", "q": "大促期间时效有变化吗？",
     "expected_source": "配送说明.md", "must_include": ["1-2 天", "顺延"], "must_not": [],
     "expect_tool": "search_kb", "note": "大促整体顺延 1-2 天"},

    # ---------- 常见问题 FAQ ----------
    {"category": "发票", "q": "电子发票多久能开？",
     "expected_source": "常见问题.md", "must_include": ["24 小时"], "must_not": [],
     "expect_tool": "search_kb", "note": "订单完成后 24h 内自动发邮箱"},
    {"category": "发票", "q": "怎么开增值税专用发票？",
     "expected_source": "常见问题.md", "must_include": ["3 个工作日"], "must_not": [],
     "expect_tool": "search_kb", "note": "专票开具周期 3 个工作日"},
    {"category": "地址", "q": "订单已发货还能改地址吗？",
     "expected_source": "常见问题.md", "must_include": ["无法修改", "不能修改", "无法改"], "must_not": [],
     "expect_tool": "search_kb", "note": "已发货无法改地址"},
    {"category": "发货", "q": "一般下单后多久发货？",
     "expected_source": "常见问题.md", "must_include": ["48 小时"], "must_not": [],
     "expect_tool": "search_kb", "note": "支付后 48 小时内发货"},
    {"category": "账户", "q": "忘记密码怎么登录？",
     "expected_source": "常见问题.md", "must_include": ["短信验证码", "手机号"], "must_not": [],
     "expect_tool": "search_kb", "note": "手机号+短信验证码登录"},
    {"category": "账户", "q": "一个账号能几台设备同时登录？",
     "expected_source": "常见问题.md", "must_include": ["3 台", "3台"], "must_not": [],
     "expect_tool": "search_kb", "note": "最多 3 台同时在线"},
    {"category": "账户", "q": "注销账号后还能恢复吗？",
     "expected_source": "常见问题.md", "must_include": ["15 天"], "must_not": ["永久", "7天", "七天"],
     "expect_tool": "search_kb", "note": "注销后 15 天可恢复，超期永久删除——别混淆"},

    # ---------- 订单查询（Agent 专属，验证真查订单库而非编）----------
    {"category": "订单查询", "q": "帮我查一下订单 SF202610130002 到哪了？",
     "expected_source": None, "must_include": ["机械键盘", "中通", "运输中"], "must_not": [],
     "expect_tool": "query_order", "note": "验 Agent 真查订单库、返回正确商品与物流"},
    {"category": "订单查询", "q": "SF202610130003 这个订单金额是多少？",
     "expected_source": None, "must_include": ["899"], "must_not": [],
     "expect_tool": "query_order", "note": "验金额字段正确返回"},
    {"category": "订单查询", "q": "我的订单 SF202610130001 签收了吗？",
     "expected_source": None, "must_include": ["已签收", "顺丰"], "must_not": [],
     "expect_tool": "query_order", "note": "验状态与物流字段"},
    {"category": "订单查询", "q": "查一下 SF999999999999 这个订单",
     "expected_source": None, "must_include": ["不存在", "核对"], "must_not": [],
     "expect_tool": "query_order", "note": "验不存在订单的友好处理，不编造"},
    {"category": "订单查询", "q": "我想查我的订单物流",
     "expected_source": None, "must_include": ["订单号", "提供"], "must_not": [],
     "expect_tool": None, "note": "没给单号 → Agent 应索要而非猜；RAG 无此工具会答不出（对比点）"},

    # ---------- 闲聊（Agent 应零工具、零检索）----------
    {"category": "闲聊", "q": "你好", "expected_source": None, "must_include": [],
     "must_not": [], "expect_tool": None, "note": "闲聊零工具：验证 Agent 不浪费检索 token"},
    {"category": "闲聊", "q": "谢谢你，辛苦了",
     "expected_source": None, "must_include": [], "must_not": [], "expect_tool": None,
     "note": "感谢语零工具"},
    {"category": "闲聊", "q": "今天天气不错啊",
     "expected_source": None, "must_include": [], "must_not": [], "expect_tool": None,
     "note": "无关话题零工具"},
    {"category": "闲聊", "q": "你们客服真耐心",
     "expected_source": None, "must_include": [], "must_not": [], "expect_tool": None,
     "note": "闲聊零工具"},

    # ---------- 超纲（知识库没有，应认输不编）----------
    {"category": "超纲", "q": "你们公司股票代码是多少？",
     "expected_source": None, "must_include": [], "must_not": [], "expect_unknown": True,
     "expect_tool": None, "note": "知识库无 → 应认输，不应编代码"},
    {"category": "超纲", "q": "能帮我写一首诗吗？",
     "expected_source": None, "must_include": [], "must_not": [], "expect_unknown": True,
     "expect_tool": None, "note": "非客服范围 → 应认输"},
    {"category": "超纲", "q": "你们老板叫什么名字？",
     "expected_source": None, "must_include": [], "must_not": [], "expect_unknown": True,
     "expect_tool": None, "note": "隐私/知识库无 → 应认输"},
    {"category": "超纲", "q": "怎么用你们的产品赚钱？",
     "expected_source": None, "must_include": [], "must_not": [], "expect_unknown": True,
     "expect_tool": None, "note": "营销类超纲 → 应认输"},
    {"category": "超纲", "q": "明天股市会涨吗？",
     "expected_source": None, "must_include": [], "must_not": [], "expect_unknown": True,
     "expect_tool": None, "note": "无关 → 应认输"},
]


# ============================================================
# 评测逻辑
# ============================================================
def _norm(s: str) -> str:
    return s.replace(" ", "").replace("　", "")


def recall_at_5(q: str, expected_source: str | None):
    if not expected_source:
        return "N/A"
    hits = vector_store.search(q, top_k=5)
    return any(h["source"] == expected_source for h in hits)


def source_hit(sources: list, expected_source: str | None):
    if not expected_source:
        return "N/A"
    return any(s.get("source") == expected_source for s in sources)


def is_hallucination(answer: str, must_not: list):
    if not must_not:
        return False
    a = _norm(answer)
    return any(_norm(k) in a for k in must_not)


def is_miss(answer: str, must_include: list):
    if not must_include:
        return False
    a = _norm(answer)
    return not any(_norm(k) in a for k in must_include)


_KNOWN_WORDS = ["未提及", "暂未", "联系人工", "不清楚", "无法", "没有相关信息",
                "知识库中", "建议联系", "不在", "不知道"]


def unknown_ok(answer: str):
    return any(w in answer for w in _KNOWN_WORDS)


def run_mode_stream(mode: str, q: str) -> dict:
    """流式跑一个内核，收集完整回答 + 首字延迟(TTFT) + 工具轨迹。"""
    start = time.perf_counter()
    answer, sources, steps = "", [], []
    first_ts = None
    gen = kernel_rag.answer_stream(q) if mode == "rag" else kernel_agent.run_stream(q)
    try:
        for ev in gen:
            if ev["type"] == "delta":
                if first_ts is None:
                    first_ts = time.perf_counter()
                answer += ev["text"]
            elif ev["type"] == "step":
                steps.append(ev)
            elif ev["type"] == "sources":
                sources = ev["sources"]
    except Exception as e:  # 单条失败不影响整体
        answer = f"[ERR] {e}"
    ttft = round((first_ts - start) * 1000, 1) if first_ts else None
    return {"answer": answer, "sources": sources, "steps": steps, "ttft_ms": ttft}


def llm_judge(question: str, answer: str, ref_text: str) -> str | None:
    """可选的 LLM 裁判：判断回答是否含资料外的编造事实。需 --judge 才调用。"""
    import llm_client
    prompt = (
        "你是严格的客服回答评测裁判。下面是一段官方知识库资料、用户问题、客服回答。\n"
        "请只判断：客服回答是否包含了资料中完全没有的、凭空编造的事实（幻觉）。\n"
        "只回答 YES 或 NO，不要解释。\n\n"
        f"【资料】\n{ref_text}\n\n【问题】{question}\n\n【回答】{answer}"
    )
    try:
        out = llm_client.complete([{"role": "user", "content": prompt}], temperature=0)
        return "YES" if out.strip().upper().startswith("YES") else "NO"
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", default="rag,agent", help="跑哪些内核，逗号分隔：rag,agent")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟用）")
    ap.add_argument("--category", default="", help="只跑某分类")
    ap.add_argument("--judge", action="store_true", help="开启 LLM 裁判（额外 token 成本）")
    args = ap.parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip() in ("rag", "agent")]

    config.check()                        # 启动体检：没 key 直接拦下
    vector_store.build()                  # 建库（已建则跳过）
    s = vector_store.stats()
    if s["degenerate"]:
        print("⚠️ 警告：检索已退化（块数 <= TOP_K），本评测测的是假检索系统！"
              "请先减小 CHUNK_SIZE 重建。\n")

    items = BENCH
    if args.category:
        items = [i for i in items if i["category"] == args.category]
    if args.limit:
        items = items[:args.limit]

    print(f"评测集：{len(items)} 条 | 内核：{modes} | judge：{args.judge}\n")
    records = []
    for n, item in enumerate(items, 1):
        q = item["q"]
        r5 = recall_at_5(q, item.get("expected_source"))
        rec = {"id": n, "category": item["category"], "q": q, "recall@5": r5, "note": item["note"]}
        for mode in modes:
            r = run_mode_stream(mode, q)
            hall = is_hallucination(r["answer"], item.get("must_not", []))
            miss = is_miss(r["answer"], item.get("must_include", []))
            sh = source_hit(r["sources"], item.get("expected_source"))
            # Agent 工具合规
            tool_called = len(r["steps"]) > 0
            exp_tool = item.get("expect_tool")
            if exp_tool is None:
                tool_ok = not tool_called
            else:
                tool_ok = any(st.get("tool") == exp_tool for st in r["steps"])
            unk = unknown_ok(r["answer"]) if item.get("expect_unknown") else None
            rec[mode] = {
                "ttft_ms": r["ttft_ms"], "answer": r["answer"],
                "source_hit": sh, "hallucination": hall, "miss": miss,
                "agent_tool_called": tool_called, "tool_ok": tool_ok,
                "unknown_ok": unk,
            }
            if args.judge and mode == "rag":
                ref = ""
                hits = vector_store.search(q, top_k=1)
                if hits:
                    ref = hits[0]["text"]
                rec["judge"] = llm_judge(q, r["answer"], ref)
        records.append(rec)
        # 单行进度
        line = f"[{n:02d}] {item['category']:<6} {q[:24]:<26} R5={r5}"
        for mode in modes:
            m = rec[mode]
            flag = "H" if m["hallucination"] else ("M" if m["miss"] else "✓")
            line += f" | {mode}:{m['ttft_ms']}ms {flag}"
        print(line)

    # ---------- 汇总 ----------
    def avg(nums):
        nums = [x for x in nums if x is not None]
        return round(sum(nums) / len(nums), 1) if nums else None

    summary = {"total": len(records), "modes": modes}
    for mode in modes:
        ttfts = [r[mode]["ttft_ms"] for r in records]
        hall = [r[mode]["hallucination"] for r in records]
        miss = [r[mode]["miss"] for r in records]
        sh_hits = [r[mode]["source_hit"] for r in records if r[mode]["source_hit"] != "N/A"]
        sh_rate = round(sum(sh_hits) / len(sh_hits), 3) if sh_hits else None
        # Agent 工具合规率
        tool_records = [r[mode]["tool_ok"] for r in records if r[mode]["tool_ok"] is not None]
        tool_rate = round(sum(tool_records) / len(tool_records), 3) if tool_records else None
        summary[mode] = {
            "avg_ttft_ms": avg(ttfts),
            "hallucination_rate": round(sum(hall) / len(hall), 3),
            "miss_rate": round(sum(miss) / len(miss), 3),
            "source_hit_rate": sh_rate,
            "agent_tool_compliance": tool_rate,
        }
    r5_all = [r["recall@5"] for r in records if r["recall@5"] != "N/A"]
    summary["recall@5_rate"] = round(sum(r5_all) / len(r5_all), 3) if r5_all else None

    # Agent 区分"调工具 vs 不调工具"的 TTFT（体现 RAG-as-Tool 的延迟 trade-off）
    if "agent" in modes:
        agent_with, agent_without = [], []
        for r in records:
            a = r["agent"]
            if a["agent_tool_called"]:
                agent_with.append(a["ttft_ms"])
            else:
                agent_without.append(a["ttft_ms"])
        summary["agent_ttft_with_tool_ms"] = avg(agent_with)
        summary["agent_ttft_without_tool_ms"] = avg(agent_without)

    print("\n================ 汇总 ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.judge:
        judges = [r.get("judge") for r in records if r.get("judge")]
        yes = sum(1 for j in judges if j == "YES")
        print(f"LLM 裁判判定幻觉：{yes}/{len(judges)}")

    out_path = ROOT / "eval_report.json"
    out_path.write_text(json.dumps(
        {"summary": summary, "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n明细已写入：{out_path}")


if __name__ == "__main__":
    main()
