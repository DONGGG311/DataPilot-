"""
Streamlit 前端界面 — DataPilot 聊天式数据分析

启动方式:
    streamlit run src/ui/app.py
"""

import os
import sys
import time
import traceback
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv()

from langchain_core.messages import AIMessage as LCAIMessage
from langchain_core.messages import HumanMessage as LCHumanMessage
from langchain_core.messages import ToolMessage

from src import get_db_manager, initialize
from src.agent.graph import DEFAULT_MODEL, compile_graph
from src.data.db import DuckDBManager  # 仅用于类型提示

# ================================================================
# 页面配置
# ================================================================

st.set_page_config(
    page_title="DataPilot — 智能数据分析助手",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ================================================================
# Session State 初始化
# ================================================================


def _init_session_state() -> None:
    """初始化 Streamlit session_state 中的持久化变量。"""
    defaults = {
        "chat_history": [],  # list[dict]: {"role", "content", "chart_paths", "dataframes"}
        "graph": None,  # 编译后的 LangGraph 实例
        "current_model": "",  # 当前 graph 使用的模型名
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ================================================================
# 侧边栏
# ================================================================


def render_sidebar(db_manager: DuckDBManager) -> str:
    """
    渲染侧边栏。

    Returns:
        用户选择的模型名称。
    """
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
        st.title("DataPilot")

        st.markdown("---")

        # ---- 模型选择 ----
        st.subheader("⚙️ 模型设置")
        model_options = [
            "claude-sonnet-4-6",
            "claude-opus-4-8",
            "claude-haiku-4-5",
            "claude-3-5-sonnet-20241022",
        ]
        default_model = os.getenv("MODEL_NAME", DEFAULT_MODEL)
        # 确保默认值在列表中
        if default_model not in model_options:
            default_index = 0
        else:
            default_index = model_options.index(default_model)

        selected_model = st.selectbox(
            "Claude 模型",
            options=model_options,
            index=default_index,
            help="选择用于分析推理的 Claude 模型。不同模型在速度、成本、能力上有所差异。",
        )

        st.markdown("---")

        # ---- 数据库状态 ----
        st.subheader("🗄️ 数据库状态")
        try:
            tables = db_manager.get_table_names()
            if tables:
                for tbl in tables:
                    row_count = db_manager.get_row_count(tbl)
                    st.metric(label=f"表: {tbl}", value=f"{row_count:,} 行")
            else:
                st.warning("暂无数据表")
        except Exception as e:
            st.error(f"数据库状态读取失败: {e}")

        # ---- 操作按钮 ----
        st.markdown("---")
        st.subheader("🔧 操作")

        if st.button("🔄 重新生成示例数据", use_container_width=True):
            with st.spinner("正在重新生成示例数据..."):
                try:
                    # 删除旧数据文件，触发 initialize() 重新生成
                    from src.data.sample_data import DEFAULT_DB_PATH
                    db_path = os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)
                    if Path(db_path).exists():
                        Path(db_path).unlink()
                    # 重新初始化（会重建数据库和示例数据）
                    initialize(db_path)
                    # 清除 graph 缓存，下次提问时重新编译
                    st.session_state.graph = None
                    st.success("示例数据已重新生成！")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    st.error(f"重新生成失败: {e}")

        if st.button("🗑️ 清空对话", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

        st.markdown("---")
        st.caption("💡 提示: 用自然语言描述你的分析需求，Agent 会自动规划并执行。")

    return selected_model


# ================================================================
# 图形化结果展示
# ================================================================


def _render_chart_image(chart_path: str) -> None:
    """在聊天消息中渲染图表图片。"""
    if Path(chart_path).exists():
        st.image(chart_path, use_container_width=True)
    else:
        st.warning(f"图表文件不存在: {chart_path}")


def _render_dataframe_preview(csv_str: str, max_rows: int = 10) -> None:
    """将 CSV 字符串渲染为 Streamlit 表格预览。"""
    import io

    import pandas as pd

    try:
        # 截断提示行
        import re

        hint_match = re.search(r"\n\n（共 \d+ 行[^)]*）", csv_str)
        clean_csv = csv_str[: hint_match.start()] if hint_match else csv_str

        df = pd.read_csv(io.StringIO(clean_csv))
        st.dataframe(df.head(max_rows), use_container_width=True)
        if len(df) > max_rows:
            st.caption(f"（共 {len(df)} 行，仅显示前 {max_rows} 行）")
    except Exception:
        st.text(csv_str[:500])


def _render_assistant_message(content: str, chart_paths: list[str], dataframes: dict[str, str]) -> None:
    """
    渲染一条助手消息 —— 包含文本、图表和数据表格。

    解析 content 中的特殊标记：
    - [图表已生成] /path/to/file.png  → 渲染为 st.image
    """
    # 显示文本内容（过滤掉系统标记行）
    lines = content.split("\n")
    clean_lines: list[str] = []
    inline_chart_paths: list[str] = []

    for line in lines:
        if line.startswith("[图表已生成]"):
            path = line.replace("[图表已生成]", "").strip()
            inline_chart_paths.append(path)
        elif line.startswith("[错误]") or line.startswith("[SQL错误]") or line.startswith("[Python错误]"):
            clean_lines.append(f"❌ {line}")
        elif line.startswith("[信息]"):
            clean_lines.append(f"ℹ️ {line.replace('[信息]', '').strip()}")
        else:
            clean_lines.append(line)

    clean_text = "\n".join(clean_lines)
    if clean_text.strip():
        st.markdown(clean_text)

    # 展示图表：优先使用 inline 解析的路径，其次使用 chart_paths
    all_chart_paths = inline_chart_paths + [p for p in chart_paths if p not in inline_chart_paths]
    if all_chart_paths:
        for path in all_chart_paths:
            _render_chart_image(path)

    # 展示数据表格
    if dataframes:
        for name, csv_str in dataframes.items():
            with st.expander(f"📊 数据: {name}", expanded=False):
                _render_dataframe_preview(csv_str)


# ================================================================
# 聊天区域
# ================================================================


def _run_analysis(user_question: str, graph, model_name: str) -> dict:
    """
    调用 Agent 工作流执行分析。

    Args:
        user_question: 用户输入的自然语言问题。
        graph: 编译后的 LangGraph 实例。
        model_name: 使用的模型名称（用于日志）。

    Returns:
        graph.invoke 的结果字典，包含 final_answer, chart_paths, dataframes, messages 等。
    """
    # 构建 LangChain 消息历史（取最近 30 条，避免上下文过长）
    langchain_messages = []
    for msg in st.session_state.chat_history[-30:]:
        if msg["role"] == "user":
            langchain_messages.append(LCHumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and msg.get("content"):
            langchain_messages.append(LCAIMessage(content=msg["content"]))

    # 添加当前用户问题
    langchain_messages.append(LCHumanMessage(content=user_question))

    # 调用工作流
    result = graph.invoke({"messages": langchain_messages})

    return result


def render_chat(db_manager: DuckDBManager, selected_model: str) -> None:
    """
    渲染聊天式主界面。
    """
    # ---- 标题 ----
    st.title("🤖 DataPilot — 智能数据分析助手")
    st.caption("用自然语言描述分析需求，AI Agent 自动规划、查询、计算并生成图表。")

    # ---- 检查 API Key ----
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or api_key == "your_api_key_here":
        st.error(
            "⚠️ 未检测到有效的 ANTHROPIC_API_KEY。\n\n"
            "请在项目根目录的 `.env` 文件中填入你的真实 API Key:\n"
            "```\nANTHROPIC_API_KEY=sk-ant-api03-...\n```\n"
            "获取 API Key: https://console.anthropic.com/\n\n"
            "当前使用的是 `.env.example` 中的占位符 `your_api_key_here`，请替换为真实 Key。"
        )
        st.stop()

    # ---- 编译 / 更新 Graph ----
    if (
        st.session_state.graph is None
        or st.session_state.current_model != selected_model
    ):
        with st.spinner("正在初始化 Agent 工作流..."):
            try:
                st.session_state.graph = compile_graph(db_manager, selected_model)
                st.session_state.current_model = selected_model
            except Exception as e:
                err_msg = str(e)
                if "403" in err_msg or "forbidden" in err_msg.lower():
                    st.error(
                        f"❌ 初始化 Agent 失败: API 访问被拒绝 (403 Forbidden)\n\n"
                        f"**可能原因:**\n"
                        f"1. API Key 无效或过期 → 检查 `.env` 中的 ANTHROPIC_API_KEY\n"
                        f"2. 地域网络限制 → 需要配置代理访问 Anthropic API\n\n"
                        f"**解决方法:**\n\n"
                        f"方式一 — HTTP 代理:\n"
                        f"```bash\n"
                        f"# 在终端中设置后重新启动 Streamlit\n"
                        f"export HTTPS_PROXY=http://127.0.0.1:7890\n"
                        f"streamlit run src/ui/app.py\n"
                        f"```\n\n"
                        f"方式二 — API 中继:\n"
                        f"在 `.env` 中配置:\n"
                        f"```\nANTHROPIC_BASE_URL=https://your-api-relay.com\n```\n\n"
                        f"详细说明见 `.env.example` 文件。\n\n"
                        f"原始错误: {e}"
                    )
                else:
                    st.error(f"初始化 Agent 失败: {e}")
                st.stop()

    graph = st.session_state.graph

    # ---- 显示历史消息 ----
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_message(
                    content=msg.get("content", ""),
                    chart_paths=msg.get("chart_paths", []),
                    dataframes=msg.get("dataframes", {}),
                )
            else:
                st.markdown(msg["content"])

    # ---- 输入框 ----
    if prompt := st.chat_input("输入你的数据分析问题，例如：华东区2023年哪个品类销售额最高？"):
        # 1. 显示用户消息
        with st.chat_message("user"):
            st.markdown(prompt)

        # 2. 添加到历史
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # 3. 执行分析
        with st.chat_message("assistant"):
            # 创建进度占位
            status_placeholder = st.empty()

            try:
                # 运行分析
                status_placeholder.info("🤔 正在分析问题并规划步骤...")

                result = _run_analysis(prompt, graph, selected_model)

                # 提取结果
                final_answer = result.get("final_answer", "")
                chart_paths = result.get("chart_paths", [])
                dataframes = result.get("dataframes", {})

                # 如果没有 final_answer，尝试从 messages 中提取最后的 AI 消息
                if not final_answer:
                    messages = result.get("messages", [])
                    for msg in reversed(messages):
                        if isinstance(msg, LCAIMessage) and msg.content:
                            final_answer = str(msg.content)
                            break

                if not final_answer:
                    final_answer = "分析完成，但未能生成文字结论。请查看上方的执行详情。"

                # 清除状态占位
                status_placeholder.empty()

                # 渲染结果
                _render_assistant_message(
                    content=final_answer,
                    chart_paths=chart_paths,
                    dataframes=dataframes,
                )

                # 添加到历史
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": final_answer,
                        "chart_paths": chart_paths,
                        "dataframes": dict(dataframes),
                    }
                )

            except Exception as e:
                status_placeholder.empty()
                err_str = str(e)
                if "403" in err_str or "forbidden" in err_str.lower():
                    error_msg = (
                        f"❌ API 访问被拒绝 (403 Forbidden)\n\n"
                        f"**可能原因:** API Key 无效 / 地域网络限制\n\n"
                        f"**解决办法:**\n"
                        f"1. 检查 `.env` 中 ANTHROPIC_API_KEY 是否为真实 Key（非占位符）\n"
                        f"2. 如在中国大陆使用，需配置代理 → 详见 `.env.example`\n\n"
                        f"原始错误: {e}"
                    )
                else:
                    error_msg = f"❌ 分析过程中出现错误: {e}"
                st.error(error_msg)
                st.code(traceback.format_exc(), language="python")
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )


# ================================================================
# 主入口
# ================================================================


def main() -> None:
    """Streamlit 应用主入口。"""
    _init_session_state()

    # 统一初始化（幂等：首次执行建库建表注入，后续调用直接返回已有实例）
    db_manager = initialize()

    # 渲染侧边栏并获取模型选择
    selected_model = render_sidebar(db_manager)

    # 渲染聊天主区域
    render_chat(db_manager, selected_model)


if __name__ == "__main__":
    main()
