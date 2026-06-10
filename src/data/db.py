"""
DuckDB 数据库连接管理

提供 DuckDBManager 类，封装连接生命周期、Schema 查询与 SQL 执行。

用法:
    from src.data.db import DuckDBManager

    with DuckDBManager() as db:
        schema = db.get_schema()
        df = db.run_sql("SELECT region, SUM(amount) FROM sales GROUP BY region")
"""

import logging
import os
from pathlib import Path
from types import TracebackType
from typing import Optional, Type

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

# 默认数据库路径（项目根目录 / data / sample.db）
DEFAULT_DB_PATH = str(Path(__file__).resolve().parent.parent.parent / "data" / "sample.db")


class DuckDBManager:
    """
    DuckDB 数据库管理器。

    封装了 DuckDB 连接的生命周期，支持上下文管理器协议，
    可从环境变量 DATABASE_PATH 读取数据库文件路径。

    Attributes:
        db_path: 数据库文件路径。
        conn:    DuckDB 连接对象（进入上下文后可用）。
    """

    def __init__(self, db_path: str | None = None) -> None:
        """
        初始化管理器。

        Args:
            db_path: DuckDB 数据库文件路径。
                     默认从环境变量 DATABASE_PATH 读取，若未设置则使用 ./data/sample.db。
        """
        self.db_path: str = db_path or os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)
        self.conn: Optional[duckdb.DuckDBPyConnection] = None
        logger.debug("DuckDBManager 已初始化，目标路径: %s", self.db_path)

    # ---- 上下文管理器协议 ----

    def __enter__(self) -> "DuckDBManager":
        """进入上下文时自动建立连接。"""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        """退出上下文时自动关闭连接（即使发生异常也会执行）。"""
        self.close()
        return False  # 不抑制异常

    # ---- 连接管理 ----

    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        建立 DuckDB 数据库连接。

        Returns:
            DuckDB 连接对象。

        Raises:
            RuntimeError: 连接失败时抛出。
        """
        if self.conn is not None:
            logger.debug("连接已存在，复用现有连接")
            return self.conn

        # 确保数据目录存在
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info("已连接 DuckDB: %s", self.db_path)
            return self.conn
        except Exception:
            logger.exception("连接 DuckDB 失败: %s", self.db_path)
            raise RuntimeError(f"无法连接到数据库: {self.db_path}")

    def close(self) -> None:
        """安全关闭数据库连接。"""
        if self.conn is not None:
            try:
                self.conn.close()
                logger.info("DuckDB 连接已关闭")
            except Exception:
                logger.warning("关闭 DuckDB 连接时出现异常", exc_info=True)
            finally:
                self.conn = None

    # ---- 查询接口 ----

    def run_sql(self, query: str) -> pd.DataFrame:
        """
        执行 SQL 查询并返回 pandas DataFrame。

        Args:
            query: 要执行的 SQL 语句（支持 DuckDB 方言）。

        Returns:
            查询结果的 pandas DataFrame。

        Raises:
            RuntimeError: 连接未建立或查询失败时抛出。
        """
        if self.conn is None:
            raise RuntimeError("数据库连接未建立，请先调用 connect() 或使用上下文管理器。")

        try:
            result = self.conn.execute(query)
            # DuckDB 的 execute 返回 DuckDBPyRelation，可直接转 DataFrame
            df = result.df()
            logger.debug("SQL 执行成功，返回 %d 行 × %d 列", len(df), len(df.columns))
            return df
        except Exception:
            logger.exception("SQL 执行失败: %s", query[:200])
            raise

    # ---- Schema 查询 ----

    def get_schema(self) -> str:
        """
        返回数据库中所有用户表的 CREATE TABLE 语句。

        这是 Agent 理解数据结构的关键入口 —— Agent 通过读取 DDL
        了解有哪些表、每张表有哪些列及其类型。

        Returns:
            所有表的 CREATE TABLE DDL 字符串，多表之间以换行分隔。
        """
        if self.conn is None:
            raise RuntimeError("数据库连接未建立，请先调用 connect() 或使用上下文管理器。")

        # 获取所有用户表名（排除系统表）
        tables = self.conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()

        if not tables:
            logger.warning("数据库中没有任何用户表")
            return "-- 数据库中暂无用户表"

        ddl_statements: list[str] = []
        for (table_name,) in tables:
            # 获取列信息
            columns = self.conn.execute(
                f"""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = '{table_name}'
                ORDER BY ordinal_position
                """
            ).fetchall()

            # 构建 CREATE TABLE 语句
            col_defs = ", ".join(
                f"    {col_name} {data_type}" for col_name, data_type in columns
            )
            ddl = f"CREATE TABLE {table_name} (\n{col_defs}\n);"
            ddl_statements.append(ddl)

        schema_text = "\n\n".join(ddl_statements)
        logger.debug("已获取 %d 张表的 Schema", len(tables))
        return schema_text

    def get_table_names(self) -> list[str]:
        """
        返回数据库中所有用户表的名称列表。

        Returns:
            表名字符串列表。
        """
        if self.conn is None:
            raise RuntimeError("数据库连接未建立，请先调用 connect() 或使用上下文管理器。")

        tables = self.conn.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
              AND table_type = 'BASE TABLE'
            ORDER BY table_name
            """
        ).fetchall()

        return [t[0] for t in tables]

    def get_row_count(self, table_name: str) -> int:
        """
        返回指定表的行数。

        Args:
            table_name: 表名。

        Returns:
            行数。
        """
        if self.conn is None:
            raise RuntimeError("数据库连接未建立，请先调用 connect() 或使用上下文管理器。")

        result = self.conn.execute(f'SELECT COUNT(*) FROM "{table_name}"').fetchone()
        return result[0] if result else 0


# ---- 向后兼容的模块级函数 ----
# 保留原有的函数签名，供轻量场景直接调用


def get_connection(db_path: str | None = None) -> duckdb.DuckDBPyConnection:
    """
    获取 DuckDB 连接（便利函数）。

    注意: 调用方需自行管理连接关闭。
    推荐使用 DuckDBManager 上下文管理器以获得自动清理。

    Args:
        db_path: 数据库文件路径，默认从环境变量读取。

    Returns:
        DuckDB 连接对象。
    """
    path = db_path or os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)
    db_dir = Path(path).parent
    db_dir.mkdir(parents=True, exist_ok=True)
    return duckdb.connect(path)


def execute_query(query: str, db_path: str | None = None) -> pd.DataFrame:
    """
    执行 SQL 查询并返回 DataFrame（便利函数）。

    注意: 每次调用都会建立新连接，频繁调用请使用 DuckDBManager。

    Args:
        query: SQL 查询语句。
        db_path: 数据库文件路径。

    Returns:
        查询结果 DataFrame。
    """
    conn = get_connection(db_path)
    try:
        return conn.execute(query).df()
    finally:
        conn.close()


def get_table_info(db_path: str | None = None) -> str:
    """
    返回数据库 Schema 信息（便利函数）。

    Args:
        db_path: 数据库文件路径。

    Returns:
        所有表的 DDL 语句字符串。
    """
    with DuckDBManager(db_path) as db:
        return db.get_schema()
