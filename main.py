from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI

import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="web"), name="static")

# === Đọc API key từ biến môi trường ===
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ Biến môi trường OPENAI_API_KEY chưa được thiết lập!")

# === Kết nối SQLite DB ===
db = SQLDatabase.from_uri("sqlite:///to_khai.db")

# === Khởi tạo mô hình GPT-4 từ langchain_openai ===
llm = ChatOpenAI(
    temperature=0,
    model="gpt-4",
    api_key=api_key  # ✅ key truyền trực tiếp
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent_executor = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("web/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())

@app.get("/ask")
async def ask(request: Request):
    query = request.query_params.get("q", "")
    try:
        result = agent_executor.run(query)
        return {"question": query, "answer": result}
    except Exception as e:
        return {"error": str(e)}
