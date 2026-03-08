import os
from sqlalchemy import create_engine, inspect
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 指向 Docker 中的 PostgreSQL
DB_URI = "postgresql://odoo:odoo@localhost:5432/postgres"
CHROMA_PATH = "./chroma_db_storage"


def ingest_to_vector_db():
    print("[IO] 连接 PostgreSQL 扫描全量表结构...")
    engine = create_engine(DB_URI)
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    docs = []
    for table in table_names:
        # 获取表内所有字段及数据类型
        columns = inspector.get_columns(table)
        col_defs = [f"{c['name']} ({c['type']})" for c in columns]

        # 极简 DDL 格式，大幅降低模型 Token 消耗
        schema_text = f"Table: {table}\nColumns: {', '.join(col_defs)}"
        docs.append(Document(page_content=schema_text, metadata={"table_name": table}))

    print(f"[引擎] 共提取 {len(docs)} 张表，开始向量化写入硬盘...")

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name="enterprise_schemas"
    )
    print("[引擎] 向量库持久化完成。")


if __name__ == "__main__":
    ingest_to_vector_db()