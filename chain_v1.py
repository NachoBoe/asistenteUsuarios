
from dotenv import load_dotenv
from pyprojroot import here
import os
from uuid import uuid4
from langsmith import Client
from langchain.docstore.document import Document

import csv
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_xml_agent, create_tool_calling_agent
from langchain.prompts import PromptTemplate, MessagesPlaceholder
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch


from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import BaseTool, StructuredTool, tool
from langchain_core.utils.function_calling import convert_to_openai_function

import unicodedata
import dill as pickle
import tiktoken
from langchain_community.vectorstores.azure_cosmos_db import AzureCosmosDBVectorSearch

from azure.search.documents.models import VectorizedQuery
from azure.search.documents.models import QueryType, QueryCaptionType, QueryAnswerType
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient

## INICIAR LANGSMITH Y API KEYS

dotenv_path = "./.env"
load_dotenv(dotenv_path=dotenv_path)


client = Client()

unique_id = uuid4().hex[0:8]
os.environ["LANGCHAIN_PROJECT"] = f"USUARIOS - {unique_id}"


# LEVANTAR DATOS
def remover_tildes(input_str):
    normalized_str = unicodedata.normalize('NFD', input_str)
    return ''.join(c for c in normalized_str if unicodedata.category(c) != 'Mn')

embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")


index_name = "usuarios2"
credential = AzureKeyCredential(os.environ["AZURE_AI_SEARCH_API_KEY"])
endpoint = "https://bantotalsearchai.search.windows.net"
search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)



# DEFINIR TOOLS

class screen_class(BaseModel):
    screen_code : str = Field(description="Screen code. It is an alphanumeric identifier that usually starts with 'H' or 'W'")
    system: str = Field(description="Sistem which the method belongs to", default="")

class description_class(BaseModel):
    description: str = Field(description="Description of the functionality of the screen.")

class systems_class(BaseModel):
    system: str = Field(description="System to retrieve the screens.")


@tool("screen_from_description", args_schema=description_class)
def screen_from_description(description:str):
    """ Semantic Search engine that retrieves top 5 screen whose functionality match the description. Retrieves the screen code, description, and the system it belongs to. """

    embedding = embeddings.embed_query(description)
    vector_query = VectorizedQuery(vector=embedding, k_nearest_neighbors=50, fields="description_vector", exhaustive=True)


    results = search_client.search(  
        search_text=description,  
        vector_queries=[vector_query],
        select=["id", "descripcion","codigo_panel", "sistema"],
        query_type=QueryType.SEMANTIC, semantic_configuration_name='my-semantic-config', query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
        top=10,
        # filter="search.in(system, )",
    )

    ans = ""
    for result in results:
        ans += f"Código de Panel (Screen code): {result['codigo_panel']} \n"
        ans += f"Sistema: {result['sistema']} \n"
        ans += f"Descripcion: {result['descripcion']} \n"
        ans += "\n\n"
    return ans



@tool("step_by_step_of_screen", args_schema=screen_class)
def step_by_step_of_screen(screen_code:str, system:str= ""):
    """ Semantic Search engine that retrieves the step by step of how to use the screen, provided the screen (panel) code and the system it belongs to. """

    filter = f"codigo_panel eq '{screen_code.upper()}'"
    if system != "":
        filter += f" and sistema eq '{system}'"
    results = search_client.search(  
        search_text= f"Quiero saber sobre el panel {screen_code} del sistema {system}",  
        select=["id", "contenido","sistema","codigo_panel" ],
        query_type=QueryType.SEMANTIC, semantic_configuration_name='my-semantic-config', query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
        top=1,
        filter=filter,
    )

    ans = ""
    for result in results:
        ans += f"Panel (screen): {result['codigo_panel']} \n"
        ans += f"Sistema: {result['sistema']} \n"
        ans += f"Información: {result['contenido']} \n"
        ans += "\n\n"
    if ans == "":
        ans = "No se encontraron resultados para la pantalla solicitada."
    return ans


@tool("screens_from_system", args_schema=systems_class)
def screens_from_system(system:str):
    """ Returns the list of all screens from a system. For each of them retrieves the screen code and a brief description of it functionality."""

    filter = f"sistema eq '{system}'"
    results = search_client.search(  
        search_text= f"",  
        select=["id", "descripcion","sistema","codigo_panel" ],
        query_type=QueryType.SEMANTIC, semantic_configuration_name='my-semantic-config', query_caption=QueryCaptionType.EXTRACTIVE, query_answer=QueryAnswerType.EXTRACTIVE,
        filter=filter,
    )

    ans = ""
    for result in results:
        ans += f"Panel (screen): {result['codigo_panel']} \n"
        ans += f"Descripcion: {result['descripcion']} \n"
        ans += "\n\n"
    return ans


def count_tokens(text: str) -> int:
    tokenizer = tiktoken.get_encoding("cl100k_base")
    tokens = tokenizer.encode(text)
    return len(tokens)


tools = [screen_from_description, step_by_step_of_screen, screens_from_system]


# DEFINIR AGENTE

openai = ChatOpenAI(model="gpt-3.5-turbo",temperature=0.0,streaming=True)


chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """
        Task: You are a helpful assistant, expert on Bantotal screens. Bantotal is a company that provides software for financial institutions. It provides numerous systems, where each of them provides a solution for different needs. The software is accessible via screen, where each system has a set of screens. Screens are uniquelly identified by a screen code, which is an alphanumeric code that usually starts with 'H' or 'W'.
        You must help the user navigate through the screens, providing information about the system and its functionalities. 
        
         INSTRUCTIONS:
        1) You must answer the user questions IN SPANISH. 
        2) You have access to basic knowledge about all the screens. This basic knowledge includes the scren code, which system it belongs to, a description of it utility and a detailed step to step on how to use it.
        3) YOU MUST NEVER MAKE INFORMATION UP. All information must be grounded over the retrieved context. 
        
        The list of systems available are:
        <systems>
            administracion_de_seguros
            alerta_lavado_de_dinero
            cadena_de_cierre
            carpeta_digital
            conciliaciones
            contabilidad
            contrapartes
            control_dual
            facultades_y_poderes
            gestion_de_eventos_de_negocio
            ingreso_de_operaciones
            matriz_de_riesgo
            mensajeria
            precios
            seguridad
            descuentos
            garantias_otorgadas
            garantias_recibidas
            gestion_de_cobranzas
            limites_de_creditos
            partners
            ahorro_programado
            cajas
            camara
            cash_management
            cofres_de_seguridad
            depositos_a_plazo
            tarjeta_de_debito
            tesoreria
        </systems>
"""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
) 


agent = create_openai_tools_agent(openai, tools, chat_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

