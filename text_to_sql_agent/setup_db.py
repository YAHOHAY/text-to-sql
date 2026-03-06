import sqlite3

def init_practice_db():
    # 在当前目录生成一个名为 practice.db 的物理文件
    conn = sqlite3.connect("practice.db")
    cursor = conn.cursor()

    print("[物理引擎] 正在构建真实的测试数据库...")

    # 1. 创建一张用户表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            vip_level INTEGER DEFAULT 0
        )
    """)

    # 2. 创建一张订单流水表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS orders (
            order_id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            amount DECIMAL(10, 2),
            order_date DATE,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    # 3. 注入一些真实的测试数据 (如果表是空的)
    cursor.execute("SELECT COUNT(*) FROM users")
    if cursor.fetchone()[0] == 0:
        cursor.executemany("INSERT INTO users (name, age, vip_level) VALUES (?, ?, ?)", [
            ("Alice", 28, 1),
            ("Bob", 35, 0),
            ("Charlie", 42, 2)
        ])
        cursor.executemany("INSERT INTO orders (user_id, amount, order_date) VALUES (?, ?, ?)", [
            (1, 150.50, "2026-03-01"),
            (1, 200.00, "2026-03-05"),
            (3, 999.99, "2026-03-06")
        ])
        conn.commit()
        print("[物理引擎] 测试数据注入完成！(包含 Alice, Bob, Charlie 的消费记录)")
    else:
        print("[物理引擎] 数据库已存在，跳过数据注入。")

    conn.close()

if __name__ == "__main__":
    init_practice_db()