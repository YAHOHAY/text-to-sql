# 使用极其精简的 Python 3.11 官方镜像作为底层系统
FROM python:3.11-slim

# 将容器内的工作目录切换到 /app
WORKDIR /app

# 先将依赖清单复制进容器
COPY requirements.txt .
# 新增这一行：强行指定去 pytorch 官方的 CPU 专属仓库拉取底座，切断 CUDA 驱动的下载
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
# 在容器内执行安装，--no-cache-dir 用于清理缓存缩小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 将宿主机当前目录的所有业务代码复制进容器的 /app 目录
COPY . .

# 声明容器内部开放 8000 端口
EXPOSE 8000

# 容器启动时执行的物理指令：用 uvicorn 拉起 FastAPI，绑定到所有网卡
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]