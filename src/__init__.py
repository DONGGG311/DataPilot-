"""
DataPilot — 智能数据分析助手

应用初始化模块，提供统一的初始化入口，供 CLI 和 Streamlit 共用。

用法:
    from src import initialize, get_db_manager

    db = initialize()          # 初始化所有依赖，返回 DuckDBManager
    db = get_db_manager()      # 获取已初始化的数据库实例
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# 在模块导入时立即加载 .env 文件
load_dotenv()

logger = logging.getLogger(__name__)

# 全局单例
_db_manager = None
_initialized = False


def initialize(db_path: str | None = None) -> "DuckDBManager":
    """
    初始化 DataPilot 所有全局依赖。

    执行顺序:
    1. 加载环境变量（已在模块导入时完成）
    2. 创建并连接 DuckDBManager
    3. 确保示例数据就绪（如不存在则自动生成）
    4. 将数据库实例注入 tools 模块（Agent 工具可调用）
    5. 返回 DuckDBManager 实例

    Args:
        db_path: 数据库文件路径，默认从 DATABASE_PATH 环境变量读取。

    Returns:
        已连接并就绪的 DuckDBManager 实例。

    Raises:
        RuntimeError: 初始化失败时抛出。
    """
    global _db_manager, _initialized

    # 避免重复初始化（但允许切换数据库路径后重新初始化）
    if _initialized and db_path is None:
        logger.debug("已初始化，跳过重复调用")
        return _db_manager

    # ---- 延迟导入避免循环依赖 ----
    from src.agent.tools import set_db_manager
    from src.data.db import DuckDBManager
    from src.data.sample_data import DEFAULT_DB_PATH, generate_sample_data

    # 确定数据库路径
    resolved_path = db_path or os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)

    logger.info("正在初始化 DataPilot...")
    logger.info("  数据库路径: %s", resolved_path)

    try:
        # 1. 创建并连接数据库
        db_manager = DuckDBManager(resolved_path)
        db_manager.connect()

        # 2. 确保示例数据存在
        db_dir = Path(resolved_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

        if not Path(resolved_path).exists():
            logger.info("  数据库文件不存在，正在生成示例数据...")
            generate_sample_data(resolved_path)
        else:
            # 检查数据表是否存在
            try:
                tables = db_manager.get_table_names()
                if not tables or db_manager.get_row_count("sales") == 0:
                    logger.info("  数据库为空，正在生成示例数据...")
                    # 需要关闭当前连接再生成（DuckDB 单文件锁）
                    db_manager.close()
                    generate_sample_data(resolved_path)
                    db_manager.connect()
            except Exception:
                logger.info("  数据库可能损坏，尝试重新生成...")
                db_manager.close()
                generate_sample_data(resolved_path)
                db_manager.connect()

        # 3. 注入到工具模块
        set_db_manager(db_manager)

        # 4. 保存全局引用
        _db_manager = db_manager
        _initialized = True

        logger.info("DataPilot 初始化完成 ✓")
        return db_manager

    except Exception:
        logger.exception("DataPilot 初始化失败")
        raise RuntimeError("初始化失败，请检查日志获取详情。")


def get_db_manager() -> "DuckDBManager":
    """
    获取已初始化的全局 DuckDBManager 实例。

    必须先调用 initialize() 完成初始化。

    Returns:
        DuckDBManager 实例。

    Raises:
        RuntimeError: 尚未初始化时抛出。
    """
    if _db_manager is None:
        raise RuntimeError(
            "DataPilot 尚未初始化。请先调用 src.initialize()。"
        )
    return _db_manager


def is_initialized() -> bool:
    """返回 DataPilot 是否已完成初始化。"""
    return _initialized and _db_manager is not None
