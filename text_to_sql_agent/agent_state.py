from typing import TypedDict, Optional, List


class SQLAgentState(TypedDict):
    """
    这是流淌在整个 LangGraph 血液里的中央账本。
    每一个节点都只能读取或修改这里面的字段。
    """
    # 1. 原始意图
    user_question: str  # 人类提出的问题 (例如: "谁是消费最高的 VIP？")

    # 2. 思考草稿
    generated_sql: str  # 大模型写出的 SQL 语句

    # 3. 物理执行结果 (事实真相)
    sql_result: Optional[List[dict]]  # 数据库真实跑出来的结果
    error_message: Optional[str]  # 🚨 极其重要：如果执行报错了，把底层的 Traceback 存在这里！

    # 4. 算法级防御
    retry_count: int  # 防止大模型陷入“写错 -> 报错 -> 继续写错”的死循环