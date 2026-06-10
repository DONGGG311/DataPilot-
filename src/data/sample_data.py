"""
示例数据生成器

生成模拟销售数据集，包含:
    - sales — 销售记录表（sale_id, region, category, sale_date, amount, quantity）

运行方式:
    python -m src.data.sample_data
"""

import logging
import os
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- 常量定义 ----
REGIONS = ["华东", "华南", "华北", "华中", "西南", "西北", "东北"]
CATEGORIES = ["电子产品", "家居用品", "服装鞋帽", "食品饮料", "美妆个护", "图书文具", "运动户外", "母婴玩具"]

DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "sample.db")
DEFAULT_ROW_COUNT = 5000
DATE_START = "2023-01-01"
DATE_END = "2024-12-31"


def _build_sales_dataframe(n_rows: int = DEFAULT_ROW_COUNT) -> pd.DataFrame:
    """
    构造模拟销售数据 DataFrame。

    Args:
        n_rows: 生成的行数，默认 5000。

    Returns:
        包含销售记录的 pandas DataFrame。
    """
    logger.info("正在生成 %d 行模拟销售数据...", n_rows)

    rng = np.random.default_rng(42)  # 固定种子，保证可复现

    # 日期范围：均匀分布
    dates = pd.date_range(start=DATE_START, end=DATE_END, freq="D")
    sale_dates = rng.choice(dates, size=n_rows)

    # 分类变量
    regions = rng.choice(REGIONS, size=n_rows, p=[0.20, 0.18, 0.15, 0.15, 0.12, 0.10, 0.10])
    categories = rng.choice(CATEGORIES, size=n_rows)

    # 数值变量：不同品类有不同价格区间和销量特征
    category_price_params = {
        "电子产品": (200, 8000, 80),
        "家居用品": (50, 3000, 50),
        "服装鞋帽": (30, 800, 30),
        "食品饮料": (5, 200, 10),
        "美妆个护": (20, 500, 20),
        "图书文具": (10, 150, 8),
        "运动户外": (50, 2000, 40),
        "母婴玩具": (30, 600, 25),
    }

    amounts = np.zeros(n_rows)
    quantities = np.zeros(n_rows, dtype=int)

    for cat in CATEGORIES:
        mask = categories == cat
        n_cat = mask.sum()
        lo, hi, scale = category_price_params[cat]
        # 金额服从对数正态分布，使分布更接近真实销售数据
        amounts[mask] = np.round(rng.lognormal(mean=np.log(scale), sigma=0.8, size=n_cat), 2)
        # 确保不超出品类价格范围
        amounts[mask] = np.clip(amounts[mask], lo, hi)
        # 数量：泊松分布，平均销量因品类而异
        avg_qty = max(1, int(scale / 80))
        quantities[mask] = np.clip(rng.poisson(lam=avg_qty, size=n_cat), 1, 20)

    df = pd.DataFrame(
        {
            "sale_id": range(1, n_rows + 1),
            "region": regions,
            "category": categories,
            "sale_date": sale_dates,
            "amount": amounts,
            "quantity": quantities,
        }
    )

    logger.info("数据生成完成 — %d 行, %d 列", len(df), len(df.columns))
    return df


def _create_table(conn: duckdb.DuckDBPyConnection) -> None:
    """在 DuckDB 中创建 sales 表（如不存在）。"""
    ddl = """
    CREATE TABLE IF NOT EXISTS sales (
        sale_id   INTEGER PRIMARY KEY,
        region    VARCHAR NOT NULL,
        category  VARCHAR NOT NULL,
        sale_date DATE NOT NULL,
        amount    DOUBLE NOT NULL,
        quantity  INTEGER NOT NULL
    )
    """
    conn.execute(ddl)
    logger.info("表 'sales' 已就绪（不存在则自动创建）")


def _insert_data(conn: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    """将 DataFrame 插入 sales 表，返回写入行数。"""
    # DuckDB 可直接从 DataFrame 注册并 INSERT
    conn.register("_tmp_sales", df)
    conn.execute("INSERT INTO sales SELECT * FROM _tmp_sales")
    conn.unregister("_tmp_sales")
    logger.info("已向 'sales' 表写入 %d 行数据", len(df))
    return len(df)


def _print_summary(conn: duckdb.DuckDBPyConnection) -> None:
    """打印数据库统计摘要。"""
    print("\n" + "=" * 56)
    print("  [DataPilot] 示例数据库摘要")
    print("=" * 56)

    # 总览
    row_count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    print(f"  总行数:     {row_count:,}")

    # 日期范围
    d_min, d_max = conn.execute(
        "SELECT MIN(sale_date), MAX(sale_date) FROM sales"
    ).fetchone()
    print(f"  日期范围:   {d_min} ～ {d_max}")

    # 区域分布
    print("\n  按区域分布:")
    region_rows = conn.execute(
        "SELECT region, COUNT(*) AS cnt FROM sales GROUP BY region ORDER BY cnt DESC"
    ).fetchall()
    for region, cnt in region_rows:
        bar = "█" * int(cnt / row_count * 30)
        print(f"    {region:<6s} {cnt:>5d}  {bar}")

    # 品类分布
    print("\n  按品类分布:")
    cat_rows = conn.execute(
        "SELECT category, COUNT(*) AS cnt FROM sales GROUP BY category ORDER BY cnt DESC"
    ).fetchall()
    for cat, cnt in cat_rows:
        bar = "█" * int(cnt / row_count * 30)
        print(f"    {cat:<8s} {cnt:>5d}  {bar}")

    # 金额统计
    total, avg = conn.execute(
        "SELECT ROUND(SUM(amount), 2), ROUND(AVG(amount), 2) FROM sales"
    ).fetchone()
    print(f"\n  总销售额:   ¥{total:,.2f}")
    print(f"  平均单价:   ¥{avg:,.2f}")

    print("=" * 56 + "\n")


def generate_sample_data(db_path: str | None = None) -> str:
    """
    生成示例销售数据并写入 DuckDB 数据库。

    功能:
      - 检查 sales 表是否已存在数据；若有则跳过生成。
      - 若表为空或不存在，则自动创建并填充 5000+ 行模拟数据。
      - 打印数据摘要。

    Args:
        db_path: DuckDB 数据库文件路径。
                 默认从环境变量 DATABASE_PATH 读取，若未设置则使用 ./data/sample.db。

    Returns:
        数据库文件路径。

    Raises:
        RuntimeError: 数据生成或写入失败时抛出。
    """
    if db_path is None:
        db_path = os.getenv("DATABASE_PATH", DEFAULT_DB_PATH)

    # 确保数据目录存在
    db_dir = Path(db_path).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    logger.info("数据库路径: %s", db_path)

    try:
        conn = duckdb.connect(db_path)

        # 建表（幂等）
        _create_table(conn)

        # 检查是否已有数据
        existing = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        if existing > 0:
            logger.info("sales 表已有 %d 行数据，跳过生成", existing)
            _print_summary(conn)
            conn.close()
            return db_path

        # 生成数据
        df = _build_sales_dataframe(DEFAULT_ROW_COUNT)
        _insert_data(conn, df)

        # 打印摘要
        _print_summary(conn)

        conn.close()
        logger.info("数据库连接已关闭")
        return db_path

    except Exception:
        logger.exception("生成示例数据时发生错误")
        raise RuntimeError("示例数据生成失败，请检查日志获取详情。")


# ---- CLI 入口 ----
if __name__ == "__main__":
    try:
        # Windows 兼容: 设置 stdout 为 UTF-8
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        print("\n===== DataPilot 示例数据生成器 =====\n")
        result_path = generate_sample_data()
        print(f"[OK] 示例数据库已创建: {result_path}")
    except RuntimeError as e:
        logger.error(str(e))
        sys.exit(1)
