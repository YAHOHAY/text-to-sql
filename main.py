from fastapi import FastAPI
from pydantic import BaseModel
# 导入在 sql_agent.py 中编译好的图引擎实例
from sql_agent import sql_app

# 初始化 FastAPI 应用实例
app = FastAPI()


# 定义接收前端请求的 JSON 数据结构
class QueryRequest(BaseModel):
    question: str


# 注册 POST 路由，定义 async 异步函数以支持 FastAPI 的并发处理
@app.post("/api/v1/chat")
async def chat_endpoint(request: QueryRequest):
    # 构造 LangGraph 所需的初始状态字典，读取前端传入的问题
    initial_state = {
        "user_question": request.question,
        "retry_count": 0,
        "error_message": None
    }

    # 调用 invoke 同步执行图引擎（如果节点内有耗时 I/O，后续可优化为 ainvoke）
    final_state = sql_app.invoke(initial_state)

    # 提取图流转结束后的关键状态数据，组装为字典返回给前端
    return {
        "sql": final_state.get("generated_sql"),
        "data": final_state.get("sql_result"),
        "answer": final_state.get("final_answer")
    }