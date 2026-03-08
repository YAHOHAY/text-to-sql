import sqlite3
import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import sqlglot
from sqlglot.expressions import Select
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from agent_state import SQLAgentState
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
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
# 1. 实例化本地词向量模型 (将文本转换为数值向量)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 2. 模拟企业级数据字典 (将表结构拆分为独立的 Document 对象)
table_documents = [
    Document(
        page_content="表名: users。包含用户基本信息。字段: user_id (主键), name (姓名), age (年龄), vip_level (VIP等级)。",
        metadata={"table_name": "users"}
    ),
    Document(
        page_content="表名: orders。包含用户订单流水记录。字段: order_id (主键), user_id (外键), amount (消费金额), order_date (订单日期)。",
        metadata={"table_name": "orders"}
    )
]

# 3. 将 Document 写入 Chroma 内存向量数据库
vector_store = Chroma.from_documents(
    documents=table_documents,
    embedding=embeddings,
    collection_name="enterprise_schemas"
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


def retrieve_schema_node(state: SQLAgentState):
    """新增节点：向量检索 (RAG)"""
    question = state["user_question"]
    print(f"\n[节点] RAG 检索: 正在向量空间匹配相关表结构...")

    # 基于问题与表结构的向量余弦相似度，检索最相关的 2 张表 (k=2)
    docs = vector_store.similarity_search(question, k=2)

    # 将检索到的 Document 拼接为纯文本
    retrieved_info = "\n".join([doc.page_content for doc in docs])
    print(f"[状态] 检索完毕，提取到以下表结构:\n{retrieved_info}")

    # 存入 State
    return {"relevant_schemas": retrieved_info}


def generate_sql_node(state: SQLAgentState):
    """修改原节点：基于检索结果生成 SQL"""
    question = state["user_question"]
    error = state.get("error_message")
    retry = state.get("retry_count", 0)

    # 核心变更：大模型不再读取全局 SCHEMA_INFO，而是读取 RAG 提供的 relevant_schemas
    schemas = state["relevant_schemas"]

    prompt = f"你是一个 SQLite 专家。请严格根据以下表结构写出 SQL。\n{schemas}\n问题: {question}\n\n输出要求：\n{parser.get_format_instructions()}"

    if error:
        prompt += f"\n\n注意：上一次 SQL 执行报错！\n报错信息: {error}\n请修复 SQL 并重新输出 JSON！"

    print(f"\n[节点] 生成 SQL (重试: {retry}/3)")
    response = llm.invoke(prompt)
    parsed_result = parser.invoke(response)
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
# 注册所有节点
workflow.add_node("retrieve_schema", retrieve_schema_node) # 新增注册
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("validate_sql", validate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("generate_report", generate_report_node)

# 重构执行流：起点 -> 检索表结构 -> 生成 SQL -> 校验 -> 执行 -> 汇报
workflow.add_edge(START, "retrieve_schema")
workflow.add_edge("retrieve_schema", "generate_sql")
workflow.add_edge("generate_sql", "validate_sql")
workflow.add_conditional_edges("validate_sql", route_after_validation)
workflow.add_conditional_edges("execute_sql", route_after_execution)
workflow.add_edge("generate_report", END)

sql_app = workflow.compile()

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