
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.chat_models import ChatOpenAI
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="web"), name="static")

# Load SQLite database
db = SQLDatabase.from_uri("sqlite:///to_khai.db")

# Set up LangChain agent
llm = ChatOpenAI(temperature=0, model="gpt-4", openai_api_key=os.getenv("OPENAI_API_KEY"))
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
