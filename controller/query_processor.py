
# import os
# import pprint

# from langchain.agents import  AgentExecutor, create_tool_calling_agent
# from langchain.prompts import ChatPromptTemplate
# from langchain.memory import ChatMessageHistory
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from utils.redis import getData, setData

# from langchain_groq import ChatGroq
# from langchain_core.chat_history import BaseChatMessageHistory

# from langchain_core.messages import HumanMessage, AIMessage

# from langchain_core.chat_history import InMemoryChatMessageHistory
# from langchain_community.callbacks import get_openai_callback
# from functools import wraps
# from langchain.embeddings import HuggingFaceEmbeddings
# import logging

# # Load API key
# groq_api_key = os.getenv('GROQ_API_KEY')

# # Initialize LLM with Llama-3-70B-8192
# llm = ChatGroq(
#     model="llama-3-70b-8192",
#     temperature=0,
#     api_key=groq_api_key
# )
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# async def agent_output( query ):
#         prompt_wydot_agent = ChatPromptTemplate.from_messages(
#             [
#                 (
#     "system",
#     "You are an assistant who helps the user with queries related to Wyoming Department of Transportation (WYDOT) specifications and technical documents. \
#     You must greet the user and use only the tools to answer questions specifically about WYDOT specifications, standards, and guidelines, and nothing else. \
#     Answer the user query only based on WYDOT documents and data provided. \
#     If the query contains a specified language, respond in that language; otherwise, respond in English. \
#     When passing the query to any comparison or search tools, do not modify the query; keep it exactly as provided by the user."
# ),

#                 ("placeholder", "{chat_history}"),
#                 ("human", "{query}"),
#                 ("placeholder", "{agent_scratchpad}"),
#             ],
#         )
#         tools = [specs_tool
#                  ]

#         agent = create_tool_calling_agent(llm, tools, prompt_wydot_agent)

#         memory = InMemoryChatMessageHistory(messages=[HumanMessage(content=" "), AIMessage(content=" ")])
#         if getData(sender):
#            a = getData(sender)
#            memory.messages = []
#            memory.messages = [HumanMessage(content=a.get("human")), AIMessage(content=a.get("AI"))] 


#         agent_executor = AgentExecutor(tools = tools,
#                                         return_intermediate_steps= False, 
#                                         handle_parsing_errors=True,
#                                         max_iterations= 5, 
#                                         early_stopping_method ='generate',
#                                         agent= agent)
#         agent_with_chat_history = RunnableWithMessageHistory(
#             agent_executor,
#             # This is needed because in most real world scenarios, a session id is needed
#             # It isn't really used here because we are using a simple in memory ChatMessageHistory
#             lambda session_id: memory,
#             input_messages_key="query",
#             history_messages_key="chat_history",
#         )


       
#         with get_openai_callback() as cb:

#             response = agent_with_chat_history.invoke(
#                 {
#                     "query": query,

#                 }, config={"configurable": {"session_id": "<foo>"}}
#                 )
                

            
#             history = {"human":response.get("query"), "AI":response.get("output")}
#             print(response.keys)
#             if response:
#                 setData(sender, history)
#             output = response.get("output")
            
#             return output

# async def query_processor(**request_params):
#     response_data={"result":"Please check response llm server"}
#     query=request_params['query']
#     print("query in query processor",query)
#     try:
#         return await agent_output(query=query)
     
#     except Exception as error:
#         print(error,"==============error parsing query============", Exception.__context__)
#         return response_data

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.milvus import Milvus
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# Environment variables
host = os.getenv('HOST', 'localhost')
port = os.getenv('PORT', '19530')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=groq_api_key
)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize Milvus vector store
database = Milvus(
    embeddings,
    connection_args={'host': host, 'port': port},
    collection_name='wyodotspecs',
    vector_field="embedding"
)

# Output parser
output_parser = StrOutputParser()

# Define prompt template
template = """
You are WYDOT chatbot, a polite and helpful Virtual Assistant of Wyoming Department of Transportation (WYDOT).
Answer the question from the given context. Ensure clarity, brevity, and human-like responses.
Context inside double backticks:``{context}```
Question inside triple backticks:```{question}```
If question is out of scope, answer it based on your role.
Provide answers in a complete sentence concisely, within 50 words.
JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.
"""
prompt = ChatPromptTemplate.from_template(template)

# Create the chain
chain = RunnableMap({
    "context": lambda x: database.similarity_search(x['question'], k=5),
    "question": lambda x: x['question']
}) | prompt | llm | output_parser

# Function to handle queries
def resultDocuments(query: str):
    result = chain.invoke({'question': query})
    return {"text": result}

# Updated query processor
async def query_processor(**request_params):
    query = request_params.get('query', '')
    if not query:
        return {"result": "No query provided."}

    try:
        response = resultDocuments(query)
        context = database.similarity_search(query, k=5)
        return {"response": response, "context": context}
    except Exception as error:
        print("Error processing query:", error)
        return {"result": "Failed to process the query."}


import os
from dotenv import load_dotenv

# Vector store / embeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.milvus import Milvus

# Google Gemini
import google.generativeai as genai

load_dotenv()

# --- Environment ---
host = os.getenv("HOST", "localhost")
port = os.getenv("PORT", "19530")
collection_name = os.getenv("MILVUS_COLLECTION", "wyodotspecs")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# --- Embeddings & Vectorstore ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)


def _retrieve_context(query: str, k: int = 5):
    """Retrieve top-k chunks from Milvus and combine into a single context string."""
    docs = database.similarity_search(query, k=k)
    context_text = "\n\n".join(d.page_content for d in docs if d.page_content)
    # Expose lightweight citations (page number if present)
    sources = [
        {
            "page": d.metadata.get("page"),
            "source": d.metadata.get("source") or d.metadata.get("file"),
            "preview": (d.page_content or "")[:300],
        }
        for d in docs
    ]
    return context_text, sources

def _make_parts(query: str, context_text: str, extracted_text: str, uploads: list):
    """Build Gemini multimodal parts."""
    parts = []
    # 1) Vectorstore context
    if context_text:
        parts.append({"text": f"CONTEXT (from vectorstore):\n{context_text}"})
    # 2) Any OCR/STT/doc text extracted from uploaded files
    if extracted_text:
        parts.append({"text": f"UPLOADED DOC/TEXT CONTEXT:\n{extracted_text}"})
    # 3) Raw media (image/audio/video) as bytes
    if uploads:
        for item in uploads:
            b = item.get("bytes")
            mime = item.get("mime") or "application/octet-stream"
            if not b:
                continue
            parts.append({"inline_data": {"mime_type": mime, "data": b}})
    # 4) The user’s question
    if query:
        parts.append({"text": f"QUESTION:\n{query}"})
    # 5) System-like guidance
    parts.insert(
        0,
        {
            "text": (
                "You are WYDOT’s helpful assistant. Answer ONLY using the provided "
                "context (vectorstore + uploaded materials).Ensure clarity, brevity, and human-like responses. If the answer is not "
                "contained, answer it based on your role. Be concise and clear."
            )
        },

       
    )
    return parts

def _run_gemini(parts: list, model_name: str = "gemini-2.5-flash") -> str:
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(parts)
    return (resp.text or "").strip()

def resultDocuments(query: str, extracted_text: str = "", uploads: list = None, model: str = "gemini-2.5-flash"):
    uploads = uploads or []
    context_text, sources = _retrieve_context(query)
    parts = _make_parts(query=query, context_text=context_text, extracted_text=extracted_text, uploads=uploads)
    answer = _run_gemini(parts, model_name=model)
    return {"text": answer, "sources": sources}

# --- Async entry for FastAPI calls ---
async def query_processor2(**request_params):
    query = request_params.get("query", "").strip()
    model = (request_params.get("model") or "gemini-2.5-flash").strip()
    uploads = request_params.get("uploads") or []
    extracted_text = request_params.get("extracted_text", "")

    if not query and not extracted_text and not uploads:
        return {"result": "No query or content provided."}

    try:
        reply = resultDocuments(query=query, extracted_text=extracted_text, uploads=uploads, model=model)
        return {
            "response": reply["text"],
            "sources": reply["sources"],
            "used_model": model,
        }
    except Exception as e:
        print("Error processing query:", e)
        return {"result": "Failed to process the query."}

