# IMPORTS

from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import dill as pickle

from dotenv import load_dotenv
from pyprojroot import here
import os
from uuid import uuid4
from langsmith import Client

import csv
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from chain_v1 import agent_executor as agent_executor_v1

import asyncio

app = FastAPI()

# Static HTML file handling using Jinja2
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# VARIABLES GLOBALES
ultima_respuesta = {}
ultima_pregunta = {}
chat_histories = {}
versions = {}

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("pagina_web.html", {"request": request})


@app.get("/health")
def health_check():
    return 'OK'


# FUNCIONES ASOCIADAS A JS

async def generate_data(message, tab_id):
    global ultima_respuesta
    global chat_histories
    global versions
    version = versions[tab_id]
    chat_history = chat_histories[tab_id]
    chat_history_string = ""
    for m in chat_history:
        chat_history_string += f"{m.type}: {m.content} \n"
    resp = ""
    print(version)
    print(version == "2")
    if version == "2":
        agent_executor = agent_executor_v1
    else:
        agent_executor = agent_executor_v1
    
    async for chunk in agent_executor.astream_events({"input": message, "chat_history": chat_history}, version="v1"):
        if chunk["event"] == "on_chat_model_stream":
            content = chunk["data"]["chunk"].content
            resp += content
            content_repl = content.replace("\n", "||")
            yield f"data: {content_repl}\n\n"
    ultima_respuesta[tab_id] = resp
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=resp))
    if len(chat_history) >= 6:
        chat_history = chat_history[-6:]
    chat_histories[tab_id] = chat_history
    yield "data: done\n\n"


@app.post("/send")
async def send(request: Request):
    global ultima_pregunta
    data = await request.json()
    tab_id = data['tabId']
    if tab_id not in chat_histories:
        chat_histories[tab_id] = []
    ultima_pregunta[tab_id] = data['message']
    return JSONResponse(content={'status': 'success'})


@app.get("/stream")
def stream(request: Request):
    tab_id = request.query_params['tabId']
    print(f"Respondiendo pregunta de {tab_id}")
    global ultima_pregunta
    return StreamingResponse(generate_data(ultima_pregunta[tab_id], tab_id), media_type='text/event-stream')


@app.post("/select_version")
async def select_version(request: Request):
    data = await request.json()
    select_version = data['version']
    global versions
    versions[data['tabId']] = select_version
    print("Versión actualizada a:", select_version)
    return JSONResponse(content={'status': 'success'})



@app.post("/new_chat")
async def handle_new_chat(request: Request):
    global chat_histories
    data = await request.json()
    tab_id = data['tabId']
    chat_histories[tab_id] = []
    return JSONResponse(content={'status': f'new chat for {tab_id}'})


@app.post("/feedback")
async def feedback(request: Request):
    global ultima_pregunta
    global ultima_respuesta
    data = await request.json()
    print(data)
    feedback = data['feedback']
    tab_id = data['tabId']
    positive = int(data["positive"]) * 2 - 1
    with open('feedback.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(["Pregunta", "Respuesta", "Version", "Positivo", "Comentario"])
        writer.writerow([ultima_pregunta[tab_id], ultima_respuesta[tab_id], versions[tab_id], positive, feedback])
    print([ultima_pregunta[tab_id], ultima_respuesta[tab_id], feedback])
    return JSONResponse(content={'status': 'Feedback recibido'})


# Uso de .env y ajustes de puerto
if __name__ == '__main__':
    load_dotenv()
    port = int(os.getenv('PORT', 8000))
    host = os.getenv('HOST', "127.0.0.1")
    import uvicorn
    uvicorn.run(app,host=host, port=port)