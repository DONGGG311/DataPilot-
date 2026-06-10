"""
Agent 工具函数

提供 DataPilot Agent 在分析过程中可调用的工具:
    - get_table_schema    — 获取数据库表结构（供 Agent 理解数据 schema）
    - execute_sql_query   — 执行 SQL 查询（通过 DuckDB）
    - execute_python_code — 在受限沙箱中执行 Python 代码
    - generate_chart      — 根据数据生成 matplotlib 图表

所有工具通过模块级全局变量 ``_db_manager`` 共享 DuckDBManager 实例。
使用前需调用 ``set_db_manager()`` 进行初始化。
"""

import csv
import io
import logging
import os
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
from langchain_core.tools import tool

# 延迟导入 DuckDBManager 类型（避免循环导入）
# 实际使用时通过 set_db_manager 注入实例

logger = logging.getLogger(__name__)

# ---- 全局数据库管理器 ----

_db_manager = None  # type: ignore


def set_db_manager(manager) -> None:
    """
    设置模块级共享的 DuckDBManager 实例。

    应在 Agent 工作流初始化时调用一次，之后所有工具函数共享该连接。

    Args:
        manager: DuckDBManager 实例。
    """
    global _db_manager
    _db_manager = manager
    logger.info("已注册 DuckDBManager 实例到工具模块")


def _get_db():
    """获取全局数据库管理器，若未初始化则抛出明确错误。"""
    if _db_manager is None:
        raise RuntimeError(
            "数据库管理器尚未初始化。"
            "请在 Agent 启动时调用 set_db_manager(manager) 或 src.initialize()。"
        )
    return _db_manager


def get_db_manager():
    """
    公开接口：获取当前已注入的全局数据库管理器。

    供外部模块（如 UI）在初始化后获取数据库实例，
    避免各模块自行维护数据库连接。

    Returns:
        DuckDBManager 实例。

    Raises:
        RuntimeError: 尚未调用 set_db_manager() 时抛出。
    """
    return _get_db()


# ---- 工具实现 ----

# 预定义安全的 exec 环境白名单（基础保护）
_ALLOWED_BUILTINS = {
    "abs", "all", "any", "ascii", "bin", "bool", "bytes", "callable",
    "chr", "complex", "dict", "divmod", "enumerate", "filter", "float",
    "format", "frozenset", "getattr", "hasattr", "hash", "hex", "int",
    "isinstance", "issubclass", "iter", "len", "list", "map", "max",
    "min", "next", "object", "oct", "ord", "pow", "print", "range",
    "repr", "reversed", "round", "set", "slice", "sorted", "str",
    "sum", "tuple", "type", "zip",
    # 数学相关
    "abs", "divmod", "max", "min", "pow", "round", "sum",
}
# 危险的模块名关键字黑名单
_DANGEROUS_IMPORT_PATTERNS = [
    r"\bos\b", r"\bsys\b", r"\bsubprocess\b", r"\bshutil\b",
    r"\bimportlib\b", r"\bctypes\b", r"\bsocket\b", r"\brequests\b",
    r"\burllib\b", r"\bhttp\b", r"\bftplib\b", r"\btelnetlib\b",
    r"\bsmtplib\b", r"\bpathlib\b", r"\bglob\b", r"\bfnmatch\b",
    r"\bpty\b", r"\bposix\b", r"\bsignal\b", r"\bmultiprocessing\b",
    r"\bthreading\b",
]
_DANGEROUS_CALLS = [
    r"\b__import__\s*\(", r"\beval\s*\(", r"\bexec\s*\(",
    r"\bcompile\s*\(", r"\bopen\s*\(", r"\bgetattr\s*\([^)]*__",
    r"\bdelattr\s*\(", r"\bsetattr\s*\(",
]


def _is_code_safe(code: str) -> tuple[bool, str]:
    """
    对 Python 代码执行基础安全检查。

    注意：这只是基础保护，不能防御精心构造的沙箱逃逸攻击。
    生产环境建议使用 Docker 容器或 subprocess 隔离执行。

    Args:
        code: 要检查的 Python 代码字符串。

    Returns:
        (is_safe, reason) 元组。safe=True 表示通过检查。
    """
    # 检查危险的 import
    for pattern in _DANGEROUS_IMPORT_PATTERNS:
        if re.search(pattern, code):
            return False, f"安全限制：代码中禁止使用模块 '{pattern.strip(r'\\b')}'"

    # 检查危险的函数调用
    for pattern in _DANGEROUS_CALLS:
        if re.search(pattern, code):
            return False, f"安全限制：代码中禁止调用 '{pattern}'"

    return True, ""


@tool
def get_table_schema() -> str:
    """
    获取当前数据库中所有表的 schema 信息（CREATE TABLE 语句）。

    用于让 Agent 了解数据库中有哪些表、每张表有哪些字段及其数据类型，
    从而准确编写 SQL 查询。

    Returns:
        数据库中所有表的 DDL 语句字符串。
        若数据库未初始化或查询失败，返回错误描述。
    """
    try:
        db = _get_db()
        schema = db.get_schema()
        return schema
    except RuntimeError as e:
        return f"[错误] {e}"
    except Exception as e:
        logger.exception("获取 schema 失败")
        return f"[错误] 获取数据库 schema 时发生异常: {e}"


@tool
def execute_sql_query(query: str) -> str:
    """
    在 DuckDB 数据库中执行 SQL 查询，返回前 50 行结果的 CSV 格式字符串。

    Agent 应使用此工具来：
    1. 先通过 get_table_schema 了解表结构
    2. 编写 SQL 查询
    3. 调用本工具执行
    4. 根据返回的结果（CSV 格式）进行下一步分析或生成图表

    Args:
        query: 要执行的 SQL 查询语句（支持 DuckDB 完整 SQL 方言）。

    Returns:
        成功时返回 CSV 格式字符串（前 50 行，含表头），并在末尾附上行数提示。
        失败时返回 [SQL错误] 前缀的错误详情，Agent 可据此修正 SQL 后重试。
    """
    try:
        db = _get_db()

        # 预处理：去除首尾空白，移除末尾分号
        query = query.strip()
        if query.endswith(";"):
            query = query[:-1].strip()

        if not query:
            return "[错误] SQL 查询为空，请提供有效的 SELECT 语句。"

        # 执行查询
        df = db.run_sql(query)

        if df.empty:
            return "[信息] 查询成功，但结果为空（0 行）。"

        total_rows = len(df)
        # 截断至前 50 行，避免 token 爆炸
        preview_df = df.head(50)

        # 转为 CSV 字符串
        output = io.StringIO()
        preview_df.to_csv(output, index=False, quoting=csv.QUOTE_NONNUMERIC)
        csv_str = output.getvalue()

        # 附加行数提示
        if total_rows > 50:
            hint = f"\n\n（共 {total_rows} 行，仅显示前 50 行。如需更多数据，请添加 WHERE 条件缩小范围。）"
        else:
            hint = f"\n\n（共 {total_rows} 行。）"

        logger.info("SQL 执行成功，返回 %d 行（预览 %d 行）", total_rows, len(preview_df))
        return csv_str + hint

    except RuntimeError as e:
        return f"[错误] {e}"
    except Exception as e:
        logger.warning("SQL 执行失败: %s", str(e))
        # 返回完整错误信息，帮助 Agent 自行修正
        return f"[SQL错误] {type(e).__name__}: {e}\n\n请检查 SQL 语法并重试。提示：\n- 确认表名和列名拼写正确\n- 检查是否使用了 DuckDB 支持的 SQL 语法\n- 可使用 get_table_schema 查看表结构"


@tool
def execute_python_code(code: str, context_df_csv: str = "") -> str:
    """
    在受限 Python 环境中执行数据处理代码，用于统计分析、数据清洗等。

    预装了以下库（已导入）：
    - pandas (as pd)
    - numpy (as np)
    - matplotlib.pyplot (as plt) — 但仅支持保存图片，不支持交互式显示

    内置变量：
    - df: 如果调用时传入了 context_df_csv 参数，则自动解析为 DataFrame 赋值给 df。
      如果不需要传入数据，df 为 None。

    重要限制：
    - 代码中禁止执行 import 语句导入 os/sys/subprocess 等系统模块
    - 禁止调用 open/eval/exec/compile 等危险函数
    - 代码输出请使用 print() 语句，stdout 会被捕获并返回

    Args:
        code: 要执行的 Python 代码字符串。使用 print() 输出结果。
        context_df_csv: 可选，CSV 格式字符串，将解析为 DataFrame 并赋值给 df 变量。
                       通常是 execute_sql_query 的返回结果。

    Returns:
        stdout 输出内容。若执行出错，返回 [Python错误] 前缀的异常信息。
    """
    # 安全检查
    is_safe, reason = _is_code_safe(code)
    if not is_safe:
        logger.warning("代码安全检查未通过: %s", reason)
        return f"[安全限制] {reason}"

    # 解析 context_df_csv
    df = None
    if context_df_csv and context_df_csv.strip():
        try:
            # 找到 CSV 数据部分（跳过末尾可能有的提示行）
            csv_part = context_df_csv
            # 如果包含提示行（如 "共 X 行"），截断掉
            hint_match = re.search(r'\n\n（共 \d+ 行[^)]*）', csv_part)
            if hint_match:
                csv_part = csv_part[:hint_match.start()]

            df = pd.read_csv(io.StringIO(csv_part))
            logger.debug("已从 CSV 解析 DataFrame: %d 行 × %d 列", len(df), len(df.columns))
        except Exception as e:
            return f"[错误] 无法解析传入的 CSV 数据: {e}"

    # 准备执行环境
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")  # 非交互后端
    import matplotlib.pyplot as plt

    # 受限的全局命名空间
    safe_globals = {
        "__builtins__": {
            k: v for k, v in __builtins__.items()  # type: ignore[attr-defined]
            if k in _ALLOWED_BUILTINS or k.startswith("_")
        } if isinstance(__builtins__, dict) else __builtins__,  # type: ignore[name-defined]
        "pd": pd,
        "np": np,
        "plt": plt,
        "df": df,
        "__name__": "__main__",
    }

    # 捕获 stdout 和 stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    old_stdout = sys.stdout
    old_stderr = sys.stderr

    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture

        exec(code, safe_globals, {})

        sys.stdout = old_stdout
        sys.stderr = old_stderr

        output = stdout_capture.getvalue()
        errors = stderr_capture.getvalue()

        if errors.strip():
            output += f"\n[stderr]\n{errors}"

        if not output.strip():
            output = "[信息] 代码执行完毕，但没有产生任何 print 输出。"

        return output

    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tb = traceback.format_exc()
        logger.warning("Python 代码执行异常: %s", str(e))
        return (
            f"[Python错误] {type(e).__name__}: {e}\n\n"
            f"完整 Traceback:\n{tb}\n\n"
            f"请检查代码逻辑并重试。"
        )
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr


@tool
def generate_chart(
    data_csv: str,
    chart_type: str = "bar",
    x: str = "",
    y: str = "",
    title: str = "DataPilot Chart",
) -> str:
    """
    根据数据生成 matplotlib 图表并保存为 PNG 文件。

    支持的图表类型：
    - bar：柱状图（默认）
    - line：折线图
    - scatter：散点图
    - pie：饼图（此时 y 参数忽略，x 列作为标签，数值用 y 列或自动计数）
    - histogram：直方图（此时 y 参数忽略，仅对 x 列绘制分布）

    Agent 使用流程：
    1. 通过 execute_sql_query 获取数据（CSV 格式）
    2. 将返回的 CSV 字符串传给本工具的 data_csv 参数
    3. 指定 x 列名、y 列名和图表标题

    Args:
        data_csv: CSV 格式的数据字符串（通常是 execute_sql_query 的返回值）。
        chart_type: 图表类型，可选 "bar", "line", "scatter", "pie", "histogram"。默认 "bar"。
        x: X 轴数据列名（或饼图的标签列）。
        y: Y 轴数据列名（柱状图/折线图/散点图使用；饼图和直方图忽略）。
        title: 图表标题，默认 "DataPilot Chart"。

    Returns:
        成功时返回生成的 PNG 文件绝对路径（如 /tmp/datapilot_chart_xxxxx.png）。
        失败时返回 [图表错误] 前缀的错误描述。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # ---- 解析输入数据 ----
    try:
        # 截断提示行
        csv_part = data_csv
        hint_match = re.search(r'\n\n（共 \d+ 行[^)]*）', csv_part)
        if hint_match:
            csv_part = csv_part[:hint_match.start()]

        if not csv_part.strip():
            return "[图表错误] 传入的数据为空，无法生成图表。"

        df = pd.read_csv(io.StringIO(csv_part))
        logger.info("图表数据: %d 行 × %d 列, 类型=%s", len(df), len(df.columns), chart_type)
    except Exception as e:
        return f"[图表错误] 无法解析 CSV 数据: {e}"

    # ---- 参数校验 ----
    chart_type = chart_type.lower().strip()
    valid_types = {"bar", "line", "scatter", "pie", "histogram"}
    if chart_type not in valid_types:
        return f"[图表错误] 不支持的图表类型 '{chart_type}'。支持的类型: {', '.join(sorted(valid_types))}"

    if chart_type != "pie":
        if not x or x not in df.columns:
            available = ", ".join(df.columns.tolist())
            return f"[图表错误] X 轴列 '{x}' 不存在。可用列: {available}"
    if chart_type in ("bar", "line", "scatter"):
        if not y or y not in df.columns:
            available = ", ".join(df.columns.tolist())
            return f"[图表错误] Y 轴列 '{y}' 不存在。可用列: {available}"

    # ---- 绘图 ----
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "bar":
            ax.bar(df[x].astype(str), df[y], color="steelblue", edgecolor="white")
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "line":
            ax.plot(df[x].astype(str), df[y], marker="o", color="steelblue", linewidth=2)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            plt.xticks(rotation=45, ha="right")

        elif chart_type == "scatter":
            ax.scatter(df[x], df[y], alpha=0.6, color="steelblue")
            ax.set_xlabel(x)
            ax.set_ylabel(y)

        elif chart_type == "pie":
            if y and y in df.columns:
                values = df[y]
            else:
                values = None
            labels = df[x].astype(str) if x and x in df.columns else df.iloc[:, 0].astype(str)
            ax.pie(
                values if values is not None else [1] * len(labels),
                labels=labels,
                autopct="%1.1f%%",
                startangle=90,
            )
            ax.axis("equal")

        elif chart_type == "histogram":
            ax.hist(df[x].dropna(), bins=20, color="steelblue", edgecolor="white", alpha=0.7)
            ax.set_xlabel(x)
            ax.set_ylabel("频数")

        ax.set_title(title, fontsize=14, fontweight="bold")
        fig.tight_layout()

        # 保存到临时文件
        tmp_dir = tempfile.gettempdir()
        tmp_path = os.path.join(tmp_dir, f"datapilot_chart_{os.getpid()}.png")
        fig.savefig(tmp_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        logger.info("图表已保存: %s", tmp_path)
        return f"[图表已生成] {tmp_path}"

    except Exception as e:
        plt.close("all")
        logger.warning("图表生成失败: %s", str(e))
        return f"[图表错误] {type(e).__name__}: {e}\n\n请检查数据格式和参数是否正确。"


# ---- 导出工具列表 ----
# LangGraph 使用此列表注册工具

ALL_TOOLS = [
    get_table_schema,
    execute_sql_query,
    execute_python_code,
    generate_chart,
]
