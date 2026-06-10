"""
LangGraph 工作流定义 — DataPilot Agent 核心编排

工作流架构:
    START
      │
      ▼
    planner_node      —— LLM 理解问题 → 生成分析步骤计划
      │
      ▼
    executor_node     —— LLM 选择工具 → 执行 SQL/Python/图表
      │  ▲
      │  ├─ 还有未执行步骤 ──┘  (循环)
      │
      ▼
    evaluator_node    —— LLM 评估是否完整回答
      │
      ├─ final_answer 非空 ──▶ END
      │
      └─ 回答不充分 ──▶ planner_node  (重新规划补充分析)

用法:
    from src.agent.graph import compile_graph
    from src.data.db import DuckDBManager

    with DuckDBManager() as db:
        graph = compile_graph(db)
        result = graph.invoke({"messages": [HumanMessage(content="分析销售趋势")]})
"""

import json
import logging
import os
from typing import Annotated, Literal, Optional

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from src.agent.tools import ALL_TOOLS, set_db_manager
from src.data.db import DuckDBManager

logger = logging.getLogger(__name__)

# ================================================================
# 常量
# ================================================================

DEFAULT_MODEL = "claude-3-5-sonnet-20241022"
MAX_EXECUTOR_RETRIES = 2  # 单步执行失败后的最大重试次数

# ================================================================
# Pydantic 模型 —— 用于 LLM 结构化输出
# ================================================================


class AnalysisPlan(BaseModel):
    """
    Planner 输出的分析计划。

    每个步骤应该是原子化的、可执行的自然语言描述，例如：
    "使用 execute_sql_query 查询华东区2023年各月销售额"
    """

    overview: str = Field(
        default="",
        description="用一句话概述整体分析思路",
    )
    steps: list[str] = Field(
        description="有序的分析步骤列表，每步是一个明确的可执行描述。通常 2-5 步。",
    )


class EvaluationResult(BaseModel):
    """Evaluator 输出的评估结论。"""

    is_complete: bool = Field(
        description="当前是否已有足够信息完整回答用户的原始问题",
    )
    final_answer: str = Field(
        default="",
        description="若 is_complete=True，输出完整的数据分析结论（含数据支撑）",
    )
    reason: str = Field(
        default="",
        description="若 is_complete=False，说明缺少什么信息，以及下一步应该做什么",
    )
    additional_steps: list[str] = Field(
        default_factory=list,
        description="若 is_complete=False，提供补充分析步骤（1-3 步）",
    )


# ================================================================
# Agent 状态定义
# ================================================================


class AgentState(BaseModel):
    """Agent 工作流共享状态 —— 在节点间传递和累积。"""

    messages: Annotated[list[BaseMessage], add_messages] = Field(
        default_factory=list,
        description="完整对话历史（含用户提问、LLM 响应、工具调用结果）",
    )
    plan: list[str] = Field(
        default_factory=list,
        description="当前分析步骤计划（有序列表）",
    )
    current_step: int = Field(
        default=0,
        description="当前正在执行的计划步骤索引",
    )
    dataframes: dict[str, str] = Field(
        default_factory=dict,
        description="中间结果存储 —— 名称 → CSV 字符串",
    )
    chart_paths: list[str] = Field(
        default_factory=list,
        description="已生成图表的文件绝对路径",
    )
    final_answer: str = Field(
        default="",
        description="最终分析结论；非空时表示工作流完成",
    )

    model_config = {"arbitrary_types_allowed": True}


# ================================================================
# 辅助函数
# ================================================================


def _make_llm(model_name: Optional[str] = None) -> ChatAnthropic:
    """
    创建 ChatAnthropic 实例。

    支持通过环境变量配置代理：
    - ANTHROPIC_BASE_URL  — API 中继/反代地址
    - HTTPS_PROXY / HTTP_PROXY — HTTP 代理（httpx 自动识别）

    Args:
        model_name: 模型名称；默认从环境变量 MODEL_NAME 读取，
                    回退到 DEFAULT_MODEL。
    """
    effective_model = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL)

    # 构建 kwargs，仅在有值时传入
    kwargs: dict = dict(
        model=effective_model,
        temperature=0.1,
        max_tokens=4096,
    )

    # 检查自定义 API 地址（代理/中继）
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
        logger.info("使用自定义 API 地址: %s", base_url)

    logger.info("初始化 ChatAnthropic，模型: %s", effective_model)
    return ChatAnthropic(**kwargs)


def _format_available_context(state: AgentState) -> str:
    """格式化当前可用的数据上下文（供 planner / evaluator 使用）。"""
    parts: list[str] = []

    # 已有的中间数据
    if state.dataframes:
        parts.append("### 已有数据摘要")
        for name, csv_str in state.dataframes.items():
            # 只展示前 3 行，避免 token 爆炸
            lines = csv_str.strip().split("\n")
            preview = "\n".join(lines[:4])  # 表头 + 3 行
            parts.append(f"- **{name}**: {len(lines) - 1} 行")
            parts.append(f"```\n{preview}\n```")

    # 已生成的图表
    if state.chart_paths:
        parts.append("### 已生成图表")
        for p in state.chart_paths:
            parts.append(f"- {p}")

    # 已执行的步骤
    if state.plan and state.current_step > 0:
        parts.append("### 已执行步骤")
        for i, step_desc in enumerate(state.plan[: state.current_step]):
            parts.append(f"- [✓] 步骤 {i + 1}: {step_desc}")
        for i, step_desc in enumerate(state.plan[state.current_step :], start=state.current_step):
            parts.append(f"- [ ] 步骤 {i + 1}: {step_desc}")

    return "\n".join(parts) if parts else "（尚无可用数据）"


def _parse_tool_error(result: str) -> Optional[str]:
    """检测工具返回是否为错误，若是则返回错误类型，否则返回 None。"""
    if result.startswith("[SQL错误]") or result.startswith("[错误]"):
        return "sql_error"
    if result.startswith("[Python错误]"):
        return "python_error"
    if result.startswith("[图表错误]"):
        return "chart_error"
    if result.startswith("[安全限制]"):
        return "security_error"
    return None


# ================================================================
# 节点实现
# ================================================================


def planner_node(state: AgentState, llm: ChatAnthropic) -> dict:
    """
    规划节点 —— 分析用户问题，生成分析步骤计划。

    输入: state.messages（从中提取最新用户问题）
    输出: 更新 state.plan, state.current_step
    """
    # 提取最新的用户问题
    user_question = ""
    for msg in reversed(state.messages):
        if isinstance(msg, HumanMessage) and not msg.content.startswith("[工具结果]"):
            user_question = msg.content
            break

    if not user_question:
        logger.warning("未找到用户问题，使用空字符串")
        user_question = "请分析数据"

    # 构建规划提示词
    context_str = _format_available_context(state)
    is_replan = len(state.plan) > 0  # 是否是在已有计划基础上重新规划

    system_prompt = f"""你是一个资深的数据分析师。你的任务是分析用户的问题，并将其拆解为清晰的、可执行的分析步骤。

当前数据库中的表结构可以通过 get_table_schema 工具获取。

{"注意：之前已经执行了一些分析步骤，现在需要补充分析。请基于已有的数据和结果，规划接下来的步骤。" if is_replan else "请从头开始规划完整的分析流程。"}

{context_str}

规划原则:
1. 每步必须是原子操作 —— 可映射到一个工具调用（get_table_schema / execute_sql_query / execute_python_code / generate_chart）
2. 步骤顺序合理 —— 先获取 schema，再查询数据，最后分析/绘图
3. 每个 SQL 查询尽量具体（包含 WHERE 条件），避免返回过多数据
4. 如需计算增长率、均值等统计量，优先在 SQL 中完成；复杂逻辑用 Python
5. 通常 2-5 步即可完成大多数分析任务
"""

    user_prompt = f"用户问题: {user_question}\n\n请生成分析计划。"

    try:
        structured_llm = llm.with_structured_output(AnalysisPlan)
        plan_result: AnalysisPlan = structured_llm.invoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )

        logger.info(
            "规划完成: %d 步 — %s",
            len(plan_result.steps),
            plan_result.overview[:80],
        )

        return {
            "plan": plan_result.steps,
            "current_step": 0,
            "final_answer": "",  # 清空旧的最终答案
            "messages": [
                AIMessage(
                    content=f"📋 **分析计划** ({'补充' if is_replan else '初始'})\n\n"
                    f"> {plan_result.overview}\n\n"
                    + "\n".join(f"{i+1}. {s}" for i, s in enumerate(plan_result.steps))
                )
            ],
        }

    except Exception as e:
        logger.exception("规划失败")
        return {
            "final_answer": f"规划分析步骤时出错: {e}",
            "messages": [AIMessage(content=f"规划失败: {e}")],
        }


def executor_node(state: AgentState, llm: ChatAnthropic) -> dict:
    """
    执行节点 —— 执行当前步骤（state.plan[state.current_step]）。

    流程:
    1. 将当前步骤描述 + 可用数据上下文发送给 LLM
    2. LLM 决定调用哪个工具及参数（通过 tool_choice）
    3. 实际执行工具调用
    4. 若失败，将错误反馈给 LLM 重试（最多 MAX_EXECUTOR_RETRIES 次）
    5. 将工具结果以 ToolMessage 存入 messages
    6. current_step += 1

    输出: 更新 state.current_step, state.messages, state.dataframes, state.chart_paths
    """
    if not state.plan:
        return {
            "messages": [AIMessage(content="[系统] 没有可执行的分析计划，请先规划。")],
        }

    step_idx = state.current_step
    if step_idx >= len(state.plan):
        return {}  # 所有步骤已执行完，无需操作

    step_desc = state.plan[step_idx]
    logger.info("执行步骤 %d/%d: %s", step_idx + 1, len(state.plan), step_desc)

    # 构建执行提示词
    context_str = _format_available_context(state)

    system_prompt = f"""你是一个数据分析执行引擎。你的任务是根据当前步骤的描述，选择合适的工具并调用它。

可用工具:
- **get_table_schema**: 获取数据库表结构（无需参数）
- **execute_sql_query**: 执行 SQL 查询，返回 CSV 数据
- **execute_python_code**: 执行 Python 代码进行统计分析；如需处理之前的查询结果，将 CSV 数据传入 context_df_csv 参数
- **generate_chart**: 根据 CSV 数据生成图表（需要 x, y 列名）

{context_str}

执行原则:
1. 必须调用工具，不要只输出文字描述
2. 如果需要先了解表结构但尚未获取，先调用 get_table_schema
3. SQL 查询尽量精确，使用 WHERE / GROUP BY / LIMIT 避免海量数据
4. 如需将上一次查询的结果传给 Python，使用 context_df_csv 参数
5. 生成图表时指定有意义的 x, y 列名和标题
"""

    user_prompt = f"当前步骤 ({step_idx + 1}/{len(state.plan)}): {step_desc}\n\n请调用合适的工具执行此步骤。"

    # 将工具绑定到 LLM
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # 构建消息列表（包含最近的上下文，但不包含过长的历史）
    # 取最近的系统消息 + 最近的工具交互 + 当前提示
    recent_messages = list(state.messages[-20:])  # 最近 20 条消息
    invoke_messages = [
        SystemMessage(content=system_prompt),
        *recent_messages,
        HumanMessage(content=user_prompt),
    ]

    # 执行 + 重试循环
    error_type: Optional[str] = None
    retry_count = 0

    while retry_count <= MAX_EXECUTOR_RETRIES:
        try:
            response: AIMessage = llm_with_tools.invoke(invoke_messages)

            # 检查 LLM 是否发起了工具调用
            if not response.tool_calls:
                # LLM 返回了文本而非工具调用 —— 可能是误解，要求重试
                logger.warning("LLM 未发起工具调用，内容: %s", str(response.content)[:200])
                if retry_count < MAX_EXECUTOR_RETRIES:
                    invoke_messages.append(response)
                    invoke_messages.append(
                        HumanMessage(content="请务必调用一个工具函数来执行此步骤。")
                    )
                    retry_count += 1
                    continue
                else:
                    # 放弃，直接返回文本作为结果
                    return {
                        "current_step": step_idx + 1,
                        "messages": [response],
                    }

            # 取第一个工具调用
            tool_call = response.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            logger.info("工具调用: %s(%s)", tool_name, tool_args)

            # 查找并执行实际的工具函数
            tool_func = None
            for t in ALL_TOOLS:
                if t.name == tool_name:
                    tool_func = t
                    break

            if tool_func is None:
                tool_result = f"[错误] 未知工具 '{tool_name}'，可用工具: {[t.name for t in ALL_TOOLS]}"
            else:
                try:
                    # 实际调用工具
                    tool_result = tool_func.invoke(tool_args)
                except Exception as tool_exc:
                    tool_result = f"[错误] 工具调用异常: {tool_exc}"

            # 检测结果是否为错误
            error_type = _parse_tool_error(str(tool_result))

            if error_type and retry_count < MAX_EXECUTOR_RETRIES:
                # 将错误反馈给 LLM，要求修正后重试
                logger.warning("工具返回错误 (%s)，第 %d 次重试...", error_type, retry_count + 1)
                tool_msg = ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"],
                )
                invoke_messages.append(response)
                invoke_messages.append(tool_msg)
                invoke_messages.append(
                    HumanMessage(
                        content=f"工具返回了 {error_type}。请修正参数后重试，或换一种方式完成步骤: {step_desc}"
                    )
                )
                retry_count += 1
                continue

            # ---- 成功完成 ----
            tool_msg = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"],
            )

            # 累积中间结果
            updated_dataframes = dict(state.dataframes)
            updated_chart_paths = list(state.chart_paths)

            result_key = f"step_{step_idx + 1}_{tool_name}"
            if tool_name == "execute_sql_query" and not error_type:
                updated_dataframes[result_key] = str(tool_result)
            elif tool_name == "execute_python_code" and not error_type:
                updated_dataframes[result_key] = str(tool_result)
            elif tool_name == "generate_chart" and not error_type:
                # 提取文件路径
                for line in str(tool_result).split("\n"):
                    if line.startswith("[图表已生成]"):
                        chart_path = line.replace("[图表已生成]", "").strip()
                        updated_chart_paths.append(chart_path)

            logger.info("步骤 %d 执行成功", step_idx + 1)

            return {
                "current_step": step_idx + 1,
                "messages": [response, tool_msg],
                "dataframes": updated_dataframes,
                "chart_paths": updated_chart_paths,
            }

        except Exception as e:
            logger.exception("执行步骤 %d 时 LLM 调用异常", step_idx + 1)
            if retry_count < MAX_EXECUTOR_RETRIES:
                retry_count += 1
                invoke_messages.append(
                    HumanMessage(content=f"发生了异常: {e}。请重试。")
                )
                continue
            else:
                return {
                    "current_step": step_idx + 1,
                    "messages": [
                        AIMessage(content=f"执行步骤失败（已重试 {MAX_EXECUTOR_RETRIES} 次）: {e}")
                    ],
                }

    # 不应该到达这里，但作为兜底
    return {"current_step": step_idx + 1}


def evaluator_node(state: AgentState, llm: ChatAnthropic) -> dict:
    """
    评估节点 —— 判断是否已完整回答用户问题。

    当所有计划步骤执行完毕后触发。
    如果回答不充分，生成补充步骤并触发重新规划（→ planner_node）。
    如果完成，生成 final_answer（→ END）。

    输出: 更新 state.final_answer, state.plan（如需补充）
    """
    # 提取原始用户问题
    user_question = ""
    for msg in state.messages:
        if isinstance(msg, HumanMessage) and not msg.content.startswith(("[工具结果]", "当前步骤")):
            user_question = msg.content
            break

    context_str = _format_available_context(state)

    # 收集已执行步骤中的所有工具结果
    tool_results_summary: list[str] = []
    for msg in state.messages:
        if isinstance(msg, ToolMessage):
            content = str(msg.content)
            # 截断过长内容
            if len(content) > 500:
                content = content[:500] + "\n... (截断)"
            tool_results_summary.append(content)

    results_text = "\n\n---\n\n".join(tool_results_summary[-10:])  # 最近 10 个结果

    system_prompt = f"""你是一个资深的数据分析师。你的任务是评估当前的分析结果是否足以完整回答用户的原始问题。

用户原始问题: "{user_question}"

{context_str}

执行结果摘要:
{results_text if results_text else "（无执行结果）"}

评估标准:
1. 用户的每个问题要点是否都有数据支撑
2. 数据是否足够具体（有数字、有对比）
3. 是否给出了清晰的结论而非仅罗列数据
4. 如有不足，需要补充什么分析

请输出评估结论。"""

    try:
        structured_llm = llm.with_structured_output(EvaluationResult)
        eval_result: EvaluationResult = structured_llm.invoke(
            [SystemMessage(content=system_prompt)]
        )

        if eval_result.is_complete:
            logger.info("评估: 分析完成")
            return {
                "final_answer": eval_result.final_answer,
                "messages": [
                    AIMessage(content=f"✅ **分析完成**\n\n{eval_result.final_answer}")
                ],
            }
        else:
            logger.info("评估: 不充分 — %s", eval_result.reason[:100])
            return {
                "plan": eval_result.additional_steps,
                "current_step": 0,
                "final_answer": "",  # 确保未完成
                "messages": [
                    AIMessage(
                        content=f"🔄 **补充分析**\n\n"
                        f"{eval_result.reason}\n\n"
                        + "\n".join(
                            f"{i+1}. {s}"
                            for i, s in enumerate(eval_result.additional_steps)
                        )
                    )
                ],
            }

    except Exception as e:
        logger.exception("评估失败")
        # 评估失败时，尝试用简单方式判断
        if state.dataframes or state.chart_paths:
            return {
                "final_answer": "分析已完成，但由于评估出错无法生成总结。请查看详细结果。",
                "messages": [AIMessage(content=f"评估异常: {e}")],
            }
        return {
            "final_answer": f"分析过程中出现错误: {e}",
            "messages": [AIMessage(content=f"评估异常: {e}")],
        }


# ================================================================
# 条件边逻辑
# ================================================================


def _after_executor(state: AgentState) -> Literal["executor", "evaluator"]:
    """执行节点后的路由判断。"""
    if state.final_answer:
        # 如果执行过程中已经设置了最终答案（如规划失败），直接去评估
        return "evaluator"
    if state.current_step < len(state.plan):
        return "executor"  # 继续执行下一步
    return "evaluator"  # 所有步骤完成，进入评估


def _after_evaluator(state: AgentState) -> Literal["planner", "end"]:
    """评估节点后的路由判断。"""
    if state.final_answer:
        return "end"
    return "planner"  # 需要重新规划补充分析


# ================================================================
# 图构建
# ================================================================


def build_graph(
    db_manager: DuckDBManager,
    model_name: Optional[str] = None,
) -> StateGraph:
    """
    构建 DataPilot Agent 工作流图（未编译）。

    工作流:
        START → planner → executor ⇄ executor (循环执行步骤)
                            ↓
                        evaluator → END 或 → planner (补充分析)

    Args:
        db_manager: 已初始化的 DuckDBManager 实例（需调用方管理生命周期）。
        model_name: Claude 模型名称；默认从 MODEL_NAME 环境变量读取。

    Returns:
        未编译的 StateGraph 实例。
    """
    # 注册全局数据库管理器到 tools 模块
    set_db_manager(db_manager)

    # 创建 LLM 实例
    llm = _make_llm(model_name)

    # 创建状态图
    workflow = StateGraph(AgentState)

    # 注册节点 —— 使用闭包捕获 llm
    workflow.add_node("planner", lambda state: planner_node(state, llm))
    workflow.add_node("executor", lambda state: executor_node(state, llm))
    workflow.add_node("evaluator", lambda state: evaluator_node(state, llm))

    # 注册边
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "executor")

    # 条件边: executor → executor (循环) 或 → evaluator
    workflow.add_conditional_edges(
        "executor",
        _after_executor,
        {
            "executor": "executor",
            "evaluator": "evaluator",
        },
    )

    # 条件边: evaluator → END 或 → planner (重新规划)
    workflow.add_conditional_edges(
        "evaluator",
        _after_evaluator,
        {
            "planner": "planner",
            "end": END,
        },
    )

    return workflow


def compile_graph(
    db_manager: DuckDBManager,
    model_name: Optional[str] = None,
):
    """
    构建并编译 DataPilot Agent 工作流，返回可直接调用的 graph 实例。

    Args:
        db_manager: DuckDBManager 实例。
        model_name: Claude 模型名称。

    Returns:
        编译后的 CompiledGraph 实例，可通过 .invoke() 或 .stream() 运行。

    用法:
        from src.data.db import DuckDBManager
        from src.agent.graph import compile_graph
        from langchain_core.messages import HumanMessage

        with DuckDBManager() as db:
            graph = compile_graph(db)
            result = graph.invoke({
                "messages": [HumanMessage(content="华东区2023年哪个品类销售额最高？")]
            })
            print(result["final_answer"])
    """
    workflow = build_graph(db_manager, model_name)
    compiled = workflow.compile()
    logger.info("DataPilot 工作流已编译就绪")
    return compiled
