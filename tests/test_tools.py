"""
工具函数单元测试

运行方式:
    python -m pytest tests/test_tools.py -v
"""

# TODO: 导入依赖
# import pytest
# import pandas as pd
# from src.agent.tools import execute_sql, execute_python, generate_chart


class TestExecuteSQL:
    """execute_sql 工具测试。"""

    def test_basic_select(self):
        """测试基本 SELECT 查询。"""
        # TODO: 创建临时表，执行查询，断言结果
        pass

    def test_invalid_sql(self):
        """测试无效 SQL 的错误处理。"""
        # TODO: 传入语法错误的 SQL，断言返回错误信息
        pass


class TestExecutePython:
    """execute_python 工具测试。"""

    def test_simple_computation(self):
        """测试基本 Python 计算。"""
        # TODO: 执行简单计算代码，断言返回结果
        pass

    def test_security_restriction(self):
        """测试危险操作的拦截。"""
        # TODO: 尝试执行 os.system / 文件写入等操作，断言被拦截
        pass


class TestGenerateChart:
    """generate_chart 工具测试。"""

    def test_bar_chart(self):
        """测试柱状图生成。"""
        # TODO: 传入简单数据，生成柱状图，断言文件存在且有效
        pass

    def test_invalid_chart_type(self):
        """测试无效图表类型的处理。"""
        # TODO: 传入不存在的图表类型，断言返回错误信息
        pass
