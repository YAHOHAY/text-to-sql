import os
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
import sqlglot
from sqlglot.expressions import Select
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from agent_state import SQLAgentState

load_dotenv()

# ==========================================
# 1. 物理层与模型初始化
# ==========================================
DB_URI = "postgresql://odoo:odoo@localhost:5432/postgres"
engine = create_engine(DB_URI)

# 向量库：只读模式，直连硬盘
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = Chroma(
    persist_directory="./chroma_db_storage",
    embedding_function=embeddings,
    collection_name="enterprise_schemas"
)

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0
)


class SQLOutput(BaseModel):
    sql_query: str = Field(description="纯 PostgreSQL 语法的执行语句")


parser = JsonOutputParser(pydantic_object=SQLOutput)


# ==========================================
# 2. 节点定义
# ==========================================
def retrieve_schema_node(state: SQLAgentState):
    """节点 1：RAG 检索。从 800 张表中精准召回相关度最高的 5 张表"""
    question = state["user_question"]
    print("[节点] RAG 检索: 在高维空间匹配表结构...")

    # k=5 控制上下文窗口大小
    docs = vector_store.similarity_search(question, k=5)
    retrieved_info = "\n\n".join([doc.page_content for doc in docs])

    print(f"[状态] 命中 {len(docs)} 张表。")
    return {"relevant_schemas": retrieved_info}


def generate_sql_node(state: SQLAgentState):
    """节点 2：生成 SQL。大模型只读取 RAG 过滤后的表结构"""
    question = state["user_question"]
    schemas = state["relevant_schemas"]
    error = state.get("error_message")
    retry = state.get("retry_count", 0)

    prompt = f"你是 PostgreSQL 专家。基于以下表结构写 SQL：\n{schemas}\n问题: {question}\n输出要求:\n{parser.get_format_instructions()}"

    if error:
        prompt += f"\n🚨 错误日志: {error}\n请修复后重试！"

    print(f"[节点] 生成 SQL (重试: {retry}/3)")
    response = llm.invoke(prompt)
    parsed = parser.invoke(response)
    return {"generated_sql": parsed["sql_query"]}


def validate_sql_node(state: SQLAgentState):
    """节点 3：AST 物理拦截。指定 dialect 为 postgres"""
    sql = state["generated_sql"]
    retry = state.get("retry_count", 0)
    try:
        parsed_ast = sqlglot.parse_one(sql, read="postgres")
        if not isinstance(parsed_ast, Select):
            return {"error_message": "安全拦截：禁止写操作！", "retry_count": retry + 1}
        return {"error_message": None}
    except Exception as e:
        return {"error_message": f"语法畸形: {str(e)}", "retry_count": retry + 1}


def execute_sql_node(state: SQLAgentState):
    """节点 4：跨网执行。使用 SQLAlchemy 执行真实网络请求"""
    sql = state["generated_sql"]
    retry = state.get("retry_count", 0)
    print(f"[节点] 执行 SQL: {sql}")

    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            rows = [dict(row._mapping) for row in result] if result.returns_rows else []
        print(f"[状态] 成功获取 {len(rows)} 条数据。")
        return {"sql_result": rows, "error_message": None}
    except Exception as e:
        error_msg = str(e).split('\n')[0]
        print(f"[状态] 报错: {error_msg}")
        return {"error_message": error_msg, "retry_count": retry + 1}


def generate_report_node(state: SQLAgentState):
    """节点 5：数据汇报"""
    prompt = f"问题: {state['user_question']}\n数据: {state.get('sql_result', [])}\n请简练回答。"
    response = llm.invoke(prompt)
    return {"final_answer": response.content}


# ==========================================
# 3. 路由与图纸组装
# ==========================================
def route_after_validation(state: SQLAgentState) -> str:
    if state.get("error_message"):
        return "generate_sql" if state.get("retry_count", 0) < 3 else END
    return "execute_sql"


def route_after_execution(state: SQLAgentState) -> str:
    if state.get("error_message"):
        return "generate_sql" if state.get("retry_count", 0) < 3 else END
    return "generate_report"


workflow = StateGraph(SQLAgentState)
workflow.add_node("retrieve_schema", retrieve_schema_node)
workflow.add_node("generate_sql", generate_sql_node)
workflow.add_node("validate_sql", validate_sql_node)
workflow.add_node("execute_sql", execute_sql_node)
workflow.add_node("generate_report", generate_report_node)

workflow.add_edge(START, "retrieve_schema")
workflow.add_edge("retrieve_schema", "generate_sql")
workflow.add_edge("generate_sql", "validate_sql")
workflow.add_conditional_edges("validate_sql", route_after_validation)
workflow.add_conditional_edges("execute_sql", route_after_execution)
workflow.add_edge("generate_report", END)

sql_app = workflow.compile()

if __name__ == "__main__":
    print("\n--- 启动测试 ---")
    # Odoo 核心用户表为 res_users
    test_question = "查询系统中所有的登录用户名及其激活状态。"

    initial_state = {
        "user_question": test_question,
        "retry_count": 0,
        "error_message": None
    }

    final_state = sql_app.invoke(initial_state)
    print(f"\n最终回答: {final_state.get('final_answer')}")