"""
config.py —— 全项目唯一的配置入口

为什么需要它（小型公司真实业务的视角）：
  1. 密钥永远不进代码 —— .env 在 .gitignore 里，代码库随便开源/给人看
  2. 配置集中一处 —— 将来换模型、换供应商，只改这里 + .env，业务代码零改动
  3. 启动即体检 —— 配置错了在第一时间报出来，而不是等到用户提问时才炸

用法（其他文件只写这一句）：
  from config import settings
"""

import os

from dotenv import load_dotenv

# 把 .env 文件里的键值对加载进环境变量（只影响当前进程）
load_dotenv()


class Settings:
    """集中管理所有可变配置。不可变的代码逻辑不写在这里。"""

    # ---- 大模型 ----
    # 千问的 OpenAI 兼容层地址：用 openai SDK 的代码，打到千问的服务器上
    BASE_URL: str = os.getenv(
        "BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    # 从环境变量读密钥。读不到就是空串，下面的体检会拦下来
    API_KEY: str = os.getenv("DASHSCOPE_API_KEY", "")
    CHAT_MODEL: str = os.getenv("MODEL", "qwen-plus")          # 对话模型
    EMBED_MODEL: str = "text-embedding-v3"                     # 向量化模型
    EMBED_DIM: int = 1024                                       # 该模型的向量维度

    # ---- 服务 ----
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    # ---- 知识库 ----
    DATA_DIR: str = "data"             # 语料目录（.md 文件们）
    CHROMA_DIR: str = "chroma_db"      # 向量库存哪（相对项目根目录）
    COLLECTION: str = "kefu_docs"      # 集合名（Chroma 要求 >= 3 字符，真踩过）
    CHUNK_SIZE: int = 400              # 每块最多多少字
    CHUNK_OVERLAP: int = 80            # 相邻块重叠多少字（防止一句话被切腰）
    TOP_K: int = 3                     # 每次检索取几块送给模型

    # ---- 客服人设（第 2 步就会用上）----
    SYSTEM_PROMPT: str = (
        "你是「星辰科技」的客服小助手，名字叫小星。"
        "你的职责：解答客户关于产品使用、订单、退换货政策、售后服务的问题。"
        "要求：语气亲切、回答简洁；不确定的问题如实告知，不要编造。"
    )


# ---- 启动体检：配置错误在这里拦截，绝不带病上岗 ----
def check() -> None:
    problems = []
    if not settings.API_KEY:
        problems.append("DASHSCOPE_API_KEY 未设置（检查 .env 文件）")
    if settings.EMBED_DIM <= 0:
        problems.append("EMBED_DIM 必须为正")
    if len(settings.COLLECTION) < 3:
        problems.append("Chroma 集合名至少 3 个字符（1.5+ 版本的硬性要求）")
    if problems:
        raise SystemExit("配置体检未通过：\n  - " + "\n  - ".join(problems))


settings = Settings()

if __name__ == "__main__":
    # 直接运行本文件 = 只做体检
    check()
    print("配置体检通过 ✓")
    print(f"  模型    : {settings.CHAT_MODEL}")
    print(f"  服务    : http://{settings.HOST}:{settings.PORT}")
    print(f"  密钥    : {'已加载（' + str(len(settings.API_KEY)) + ' 位）' if settings.API_KEY else '未设置'}")
