# 星辰客服小助手 · kefu-agent

一个能跑、能看、能对账的**智能客服 Agent**：FastAPI + Function Calling + RAG + SQLite 会话持久化 + SSE 流式前端。

按「**垂直切片**」方式从 0 长出来：每一步都切成一条能端到端跑通、能亲眼验收的细条，
先把「能对话」跑通，再一层层加会话、知识库、检索、工具调用、工程防护，最后才包前端和接口。

---

## 一、怎么跑起来

```bash
cd kefu-agent

# 1. 配密钥：项目根新建 .env（已在 .gitignore 里，绝不会进代码库）
#    DASHSCOPE_API_KEY=sk-xxxxxxxx

# 2. 装依赖（版本已钉死，一把装好）
pip install -r requirements.txt

# 3. 先做配置体检（key 没配会在这一步就报出来，而不是等到提问时才炸）
python config.py

# 4. 建库并体检（会打印 chunk 数，并告诉你检索有没有退化）
python vector_store.py

# 5. 起服务
python main.py
```

打开 **http://127.0.0.1:8000** 聊天；**http://127.0.0.1:8000/docs** 是 FastAPI 自动生成的接口文档。

---

## 二、八步垂直切片（每步都有独立验收点）

| 步 | 做的事 | 怎么验收「真的成了」 |
|---|---|---|
| 1 | 最小对话：FastAPI + `/api/chat` | 命令行 curl 一条消息，收到模型回话 |
| 2 | **会话**：SQLite 落盘 + 历史回灌 | 刷新页面对话还在；服务端重启也不丢 |
| 3 | **知识库**：`data/*.md` → 切块 → 向量化 → Chroma | `python vector_store.py` 打印块数与来源分布 |
| 4 | **RAG**：先检索再作答 + 防幻觉规则 | 问超纲问题，它如实说「知识库中暂未提及」而不是编 |
| 5 | **Agent**：Function Calling + 手写 agent loop | 问订单 → 看它真的调了 `query_order`；闲聊 → 零工具调用 |
| 6 | **防护**：轮次上限 / 结果缓存 / 重复调用熔断 | 轨迹面板里能看到 `cached: true` 和熔断收口 |
| 7 | **前端**：SSE 流式 + 工具轨迹可视化 + 输入锁 | 逐字打字效果；连发也不会串台 |
| 8 | **收口**：双内核切换 + 全链路自测 | `mode=rag` 与 `mode=agent` 各跑一遍对照 |

**为什么是垂直切片而不是先写完后端再写前端**：每一步都能跑、能看、能对账，
出 bug 时能立刻定位在「刚加的那一层」，而不是在几百行里找。

---

## 三、四层架构

| 层 | 文件 | 职责 |
|---|---|---|
| 语料 | `data/*.md` | 3 份客服语料：退换货政策 / 配送说明 / 常见问题 |
| 配置 | `config.py` | 唯一配置入口 + **启动即体检**（`check()`） |
| 底座 | `llm_client.py`<br>`vector_store.py` | 千问客户端（对话 / 向量化 / 流式）<br>切分 + 建库 + 检索 + **体检** |
| 内核 | `kernel_rag.py`<br>`kernel_agent.py` | A：RAG 单轮检索问答<br>B：手写 Function Calling agent loop |
| 服务 | `main.py` + `db.py` | FastAPI 路由 / SSE / SQLite 会话持久化 |
| 前端 | `static/index.html` | 双模式切换、工具轨迹可视化、引用来源、输入锁 |

**搭建顺序的关键**：先跑通内核（`python vector_store.py` 直接看检索结果），再包 API。
反过来会把「内核 bug」误判成「接口 bug」，排查成本翻倍。

---

## 四、两种内核：RAG 和 Agent 到底差在哪

```
RAG 内核（mode=rag）
    每问必检索 → 资料塞进 prompt → 单轮作答
    优点：稳、便宜、可溯源   缺点：闲聊也白检索一次；证据不够不会「再翻一页」

Agent 内核（mode=agent）
    模型自己决定 → 请求调工具 → 后端执行 → 结果回填 → 可能再调一次
    优点：闲聊零检索、复杂问题多步取证   缺点：多轮推理，慢且贵
```

**本项目是 RAG-as-Tool**：`_tool_search_kb()` 就是把 RAG 用的那套检索降级成一个「工具」塞进 agent loop，
检索逻辑一行没重写。真实工程里几乎都是这种混合形态，不是二选一。

---

## 五、Agent Loop 的三个工程防护（`kernel_agent.py`）

不是防御性编程炫技——是实测发现模型真会这么干才加的：

| 防护 | 实现 | 挡住的是什么 |
|---|---|---|
| 轮次上限 | `MAX_TURNS = 5` | 模型抽风无限循环，token 烧穿 |
| 工具结果缓存 | `cache[(名字, 参数)]` | 同一工具同一参数重复执行，白等一轮网络 |
| 连续重复熔断 | `repeats >= 2` 强制收口 | 模型用完全相同的 keyword 反复调同一工具，返回一模一样 |

**安全模型**：模型只能「请求」调工具（返回工具名 + 参数字符串），
真正执行靠 `FUNC_MAP` 白名单查表——**模型永远无法直接执行你的代码**。

---

## 六、检索现状与升级路径（诚实说明）

本项目的检索是**纯向量检索**（`text-embedding-v3` + Chroma），`TOP_K = 3`。
这是刻意的：先让系统跑起来，等暴露出具体失败案例再对症下药。

已知局限与下一步（已在姊妹项目 `kb-site` 中完整验证过）：

| 局限 | 表现 | 升级方案 |
|---|---|---|
| 换关键词结果不变 | 纯向量对关键词不敏感 | BM25 关键词召回 + **RRF 融合**（只用排名不用分数，规避量纲不可比） |
| 真答案卡在第 4 名 | 被 `TOP_K` 挡在门外 | `gte-rerank-v2` 交叉编码精排：粗排 10 → 精排 3 |
| 库太小 | 块数 ≤ TOP_K = 全量注入，测的是假检索 | `stats()` 的 `degenerate` 字段直接报警 |

**上手任何 RAG 的第一件事**：先跑 `python vector_store.py` 看 `stats()`。
库里只有 2 块却设 `TOP_K=3`，那是「假检索系统」，后面所有评估结论都是假的。

---

## 七、接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 聊天页面 |
| POST | `/api/chat` | 非流式问答，`mode` = `agent`（默认）/ `rag` |
| POST | `/api/chat/stream` | **SSE 流式**：`step` / `sources` / `delta` / `done` 四类事件 |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{sid}` | 某会话完整消息（前端刷新后恢复对话） |

```bash
curl -X POST http://127.0.0.1:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "我的订单 SF202610130002 到哪了", "mode": "agent"}'
```

**信任边界**：请求体里只有 `message` / `history` / `mode` / `session_id`，
**密钥永远留在服务端**。`ChatMessage.role` 用 `Literal["user","assistant"]` 约束——
前端不许自称 `system`，否则能冒充人设、篡改客服指令，这是真实攻击面。

---

## 八、踩坑记录（全部真实踩过）

| # | 现象 | 原因 | 修复 |
|---|---|---|---|
| 1 | embedding 报 `batch size is invalid` | 千问 embedding 单批上限 10 条 | `llm_client.embed` 自动分批 |
| 2 | Chroma `Validation error: name` | 1.5+ 要求集合名 ≥ 3 字符 | `"kb"` → `"kefu_docs"` |
| 3 | `shutil.rmtree` 被拦 | 容器/只读环境常见 | 改用 `chroma.delete_collection()` |
| 4 | 回填工具结果后模型「看不见」 | 用了 `tc["id"]` | tc 是 SDK 对象，要用 `tc.id` |
| 5 | Agent 重复调同一工具空转 | 模型抽风 | 缓存 + 熔断（见第五节） |
| 6 | 流式最后一个 chunk 报越界 | 尾包 `choices` 为空 | `if not chunk.choices: continue` |
| 7 | 前端连发两条消息串台 | 异步竞态 | `pending` 输入锁 + 按钮 disabled + 失败回滚 |
| 8 | Git Bash 内联中文 JSON 报 `error parsing body` | 单引号转义 | 用 `-d @req.json` 文件传参 |
| 9 | 服务启动后没人访问就白等建库 | 建库在首个请求里 | 挪到 `lifespan`，启动即建 |
| 10 | 模型超时/限流给用户甩 500 | 异常未兜底 | `try/except` 降级「系统开小差」，错误留服务端日志 |

