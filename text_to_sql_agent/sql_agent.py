import sqlite3
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import sqlglot
from sqlglot.expressions import Select
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


def validate_sql_node(state: SQLAgentState):
    """节点 1.5：AST 安全校验（纯物理逻辑隔离）"""
    sql = state["generated_sql"]
    retry = state.get("retry_count", 0)
    print(f"[节点] 安全校验: 检查是否包含危险指令...")

    try:
        # 将 SQL 解析为 AST 语法树
        parsed_ast = sqlglot.parse_one(sql)

        # 核心防御：如果这棵树的根节点不是 Select (查询)，直接拦截！
        if not isinstance(parsed_ast, Select):
            error_msg = "安全拦截：系统只允许执行 SELECT 查询语句，禁止任何写操作（INSERT/UPDATE/DELETE/DROP）！"
            print(f"[拦截] ❌ {error_msg}")
            return {"error_message": error_msg, "retry_count": retry + 1}

        print("[状态] ✅ 安全校验通过，纯查询语句。")
        return {"error_message": None}  # 校验通过，错误清空

    except Exception as e:
        error_msg = f"SQL语法严重畸形，无法解析: {str(e)}"
        print(f"[拦截] ❌ {error_msg}")
        return {"error_message": error_msg, "retry_count": retry + 1}
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
def route_after_validation(state: SQLAgentState) -> str:
    """校验后的路口：报错打回，安全则放行执行"""
    if state.get("error_message"):
        return "generate_sql" if state.get("retry_count", 0) < 3 else END
    return "execute_sql"


def generate_report_node(state: SQLAgentState):
    """节点 3：数据翻译（将查出的原始数据转为人类语言）"""
    question = state["user_question"]
    results = state.get("sql_result", [])

    # 组装提示词：包含人类的原始问题，以及刚从数据库里捞出来的真实数据
    prompt = f"用户问题: {question}\n数据库返回的真实数据: {results}\n请用简练的自然语言给出最终回答。"

    # 再次调用 LLM。这里不需要结构化输出，直接要普通文本即可
    response = llm.invoke(prompt)

    # 将生成的文本存入状态字典的 final_answer 键中
    return {"final_answer": response.content}


# ----------------------------------------
# 修改路由函数，将原先成功的终点指向新节点
# ----------------------------------------
def route_after_execution(state: SQLAgentState) -> str:
    """条件边：判断执行结果决定流转"""
    error = state.get("error_message")
    retry = state.get("retry_count", 0)

    if error:
        if retry < 3:
            print("[路由] 触发反思重试")
            return "generate_sql"
        else:
            print("[路由] 重试上限，终止")
            return END

    # 修改此处：执行成功后，不再直接 END，而是进入汇报节点
    print("[路由] 执行无误，进入数据汇报生成")
    return "generate_report"


# ----------------------------------------
# 重新组装图纸
# ----------------------------------------
workflow = StateGraph(SQLAgentState)

workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("validate_sql", validate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
# 新增：将汇报节点注册进状态图
workflow.add_node("generate_report", generate_report_node)

workflow.add_edge(START, "generate_sql")
workflow.add_edge("generate_sql", "validate_sql")
workflow.add_conditional_edges("validate_sql", route_after_validation)
workflow.add_conditional_edges("execute_sql", route_after_execution)

# 新增：汇报节点走完后，整个流程才真正结束
workflow.add_edge("generate_report", END)

sql_app = workflow.compile()
if __name__ == "__main__":
    print("\n--- 启动测试 ---")

    test_question = "帮我查一下名字叫 Charlie 的用户，他一共消费了多少钱？"

    initial_state = {
        "user_question": test_question,
        "retry_count": 0,
        "error_message": None
    }

    final_state = sql_app.invoke(initial_state)

    print("\n最终结果:")
    print(final_state.get("sql_result"))