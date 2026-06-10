"""
DataPilot — 命令行入口

用法:
    python main.py initdb           初始化数据库并生成示例数据
    python main.py run              启动 Streamlit 聊天界面
    python main.py ask "你的问题"    直接在命令行中提问（CLI 模式）
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

BANNER = r"""
  ____        _        ____  _  _  _
 |  _ \  __ _| |_ __ _|  _ \| |(_)| |_  ___  _ __
 | | | |/ _` | __/ _` | |_) | || || __|/ _ \| '__|
 | |_| | (_| | || (_| |  __/| || || |_| (_) | |
 |____/ \__,_|\__\__,_|_|   |_||_| \__\___/|_|

       智能数据分析助手 — AI Agent 驱动
"""


def cmd_initdb(args) -> int:
    """初始化数据库并生成示例数据。"""
    print("正在初始化数据库...")

    from src import initialize

    try:
        db = initialize(args.db_path)
        tables = db.get_table_names()
        if "sales" in tables:
            row_count = db.get_row_count("sales")
            print(f"  [OK] 数据库就绪: {args.db_path or os.getenv('DATABASE_PATH', './data/sample.db')}")
            print(f"  [OK] sales 表: {row_count:,} 行")
        else:
            print("  [WARN] sales 表未找到，请检查日志")
            return 1
        return 0
    except Exception as e:
        print(f"  [FAIL] 初始化失败: {e}")
        return 1


def cmd_run(args) -> int:
    """启动 Streamlit 应用。"""
    app_path = PROJECT_ROOT / "src" / "ui" / "app.py"

    if not app_path.exists():
        print(f"错误: 找不到 Streamlit 应用文件: {app_path}")
        return 1

    print("正在启动 Streamlit 应用...")
    print(f"  文件: {app_path}")
    print("  按 Ctrl+C 停止\n")

    # 使用 streamlit run 启动
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(app_path)]
        + (["--server.port", str(args.port)] if args.port else [])
    )


def cmd_ask(args) -> int:
    """CLI 模式：直接提问并获取分析结果。"""
    question = args.question

    print(f"\n问题: {question}\n")
    print("正在分析...")

    from langchain_core.messages import HumanMessage

    from src import initialize
    from src.agent.graph import compile_graph

    try:
        # 初始化
        db = initialize()

        # 编译 graph
        model = args.model or os.getenv("MODEL_NAME", None)
        graph = compile_graph(db, model)

        # 执行分析
        result = graph.invoke({"messages": [HumanMessage(content=question)]})

        # 输出结果
        final_answer = result.get("final_answer", "")
        chart_paths = result.get("chart_paths", [])

        if final_answer:
            print("\n" + "=" * 60)
            print("分析结论:")
            print("=" * 60)
            print(final_answer)
        else:
            print("\n（未生成文字结论，请查看上方工具输出）")

        if chart_paths:
            print(f"\n生成图表: {len(chart_paths)} 个")
            for p in chart_paths:
                print(f"  - {p}")

        return 0

    except Exception as e:
        print(f"\n分析失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main() -> int:
    """CLI 主入口，返回退出码。"""
    parser = argparse.ArgumentParser(
        prog="datapilot",
        description="DataPilot — 基于 AI Agent 的智能数据分析助手",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py initdb                 初始化数据库
  python main.py run                    启动 Web 界面
  python main.py run --port 8502        指定端口启动
  python main.py ask "华东区2023年哪个品类销售额最高？"
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # ---- initdb ----
    parser_initdb = subparsers.add_parser("initdb", help="初始化数据库并生成示例数据")
    parser_initdb.add_argument(
        "--db-path",
        default=None,
        help="数据库文件路径（默认从 DATABASE_PATH 环境变量读取）",
    )
    parser_initdb.set_defaults(func=cmd_initdb)

    # ---- run ----
    parser_run = subparsers.add_parser("run", help="启动 Streamlit Web 界面")
    parser_run.add_argument(
        "--port",
        type=int,
        default=None,
        help="Streamlit 服务端口（默认 8501）",
    )
    parser_run.set_defaults(func=cmd_run)

    # ---- ask ----
    parser_ask = subparsers.add_parser("ask", help="命令行直接提问分析")
    parser_ask.add_argument("question", help="数据分析问题（自然语言）")
    parser_ask.add_argument(
        "--model",
        default=None,
        help="Claude 模型名称（默认从 MODEL_NAME 环境变量读取）",
    )
    parser_ask.set_defaults(func=cmd_ask)

    # ---- 解析 ----
    if len(sys.argv) == 1:
        print(BANNER)
        parser.print_help()
        return 0

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    print(BANNER)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
