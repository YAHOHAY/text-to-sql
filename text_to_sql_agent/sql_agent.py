import sqlite3
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from agent_state import SQLAgentState

# 引入 LangChain 的 JSON 解析器，用于处理纯文本到 JSON 字典的转换
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

# 初始化 DeepSeek 客户端
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)


# 定义输出结构
class SQLOutput(BaseModel):
    sql_query: str = Field(description="能在 SQLite 中直接执行的纯 SQL 语句")


# 实例化解析器，并绑定我们定义的 SQLOutput 数据模型
parser = JsonOutputParser(pydantic_object=SQLOutput)

# 测试数据库的表结构声明
SCHEMA_INFO = """
Table: users (user_id INTEGER PRIMARY KEY, name TEXT, age INTEGER, vip_level INTEGER DEFAULT 0)
Table: orders (order_id INTEGER PRIMARY KEY, user_id INTEGER, amount DECIMAL, order_date DATE)
"""


def generate_sql_node(state: SQLAgentState):
    """节点 1：生成 SQL"""
    question = state["user_question"]
    error = state.get("error_message")
    retry = state.get("retry_count", 0)

    # parser.get_format_instructions() 获取 Pydantic 模型对应的 JSON 格式要求，拼接到提示词中
    prompt = f"你是一个 SQLite 专家。请根据表结构写出 SQL。\n{SCHEMA_INFO}\n问题: {question}\n\n输出要求：\n{parser.get_format_instructions()}"

    # 如果状态中包含错误信息，将其追加到提示词中让模型反思
    if error:
        prompt += f"\n\n注意：上一次 SQL 执行报错！\n报错信息: {error}\n请修复 SQL 并重新输出 JSON！"

    print(f"\n[节点] 生成 SQL (重试次数: {retry}/3)")

    # 调用 LLM 获取纯文本响应
    response = llm.invoke(prompt)

    # 将 LLM 返回的文本通过解析器转换为 Python 字典
    parsed_result = parser.invoke(response)

    # 提取字典中的 sql_query 字段存入状态
    return {"generated_sql": parsed_result["sql_query"]}


def execute_sql_node(state: SQLAgentState):
    """节点 2：执行 SQL"""
    sql = state["generated_sql"]
    retry = state.get("retry_count", 0)
    print(f"[节点] 执行 SQL: {sql}")

    try:
        conn = sqlite3.connect("practice.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(sql)
        rows = cursor.fetchall()
        conn.close()

        results = [dict(row) for row in rows]
        print(f"[状态] 执行成功，获取到 {len(results)} 条数据。")

        # 执行成功，清空错误信息
        return {"sql_result": results, "error_message": None}

    except Exception as e:
        error_msg = str(e)
        print(f"[状态] 执行失败，报错: {error_msg}")

        # 执行失败，记录错误信息并增加重试计数
        return {"error_message": error_msg, "retry_count": retry + 1}


def route_after_execution(state: SQLAgentState) -> str:
    """条件边：判断执行结果决定流转"""
    error = state.get("error_message")
    retry = state.get("retry_count", 0)

    if error:
        # 有错误且未达到上限，返回重试节点
        if retry < 3:
            print("[路由] 触发反思重试")
            return "generate_sql"
        else:
            print("[路由] 重试上限，终止")
            return END

    print("[路由] 执行无误，结束流转")
    return END


# 构建状态图
workflow = StateGraph(SQLAgentState)

workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)

workflow.add_edge(START, "generate_sql")
workflow.add_edge("generate_sql", "execute_sql")
workflow.add_conditional_edges("execute_sql", route_after_execution)

sql_app = workflow.compile()

if __name__ == "__main__":
    print("\n--- 启动测试 ---")

    test_question = "查一下 Charlie 的邮箱地址是多少？"

    initial_state = {
        "user_question": test_question,
        "retry_count": 0,
        "error_message": None
    }

    final_state = sql_app.invoke(initial_state)

    print("\n最终结果:")
    print(final_state.get("sql_result"))