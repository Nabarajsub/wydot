
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



# import os
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores.milvus import Milvus
# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnableMap
# from langchain.schema.output_parser import StrOutputParser

# load_dotenv()

# # --- Environment variables ---
# host = os.getenv('HOST', 'localhost')
# port = os.getenv('PORT', '19530')
# groq_api_key = os.getenv('GROQ_API_KEY')
# collection_name = os.getenv('MILVUS_COLLECTION', 'wyodotspecs')  # must match the created collection

# # --- LLM initialization ---
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",
#     temperature=0,
#     api_key=groq_api_key
# )

# # --- Embeddings ---
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     encode_kwargs={"normalize_embeddings": True}
# )

# # --- Connect to existing Milvus collection ---
# database = Milvus(
#     embeddings,
#     connection_args={'host': host, 'port': port},
#     collection_name=collection_name,
#     vector_field="embedding"
# )

# # Use as retriever for cleaner integration
# retriever = database.as_retriever(search_kwargs={"k": 5})

# # --- Output parser ---
# output_parser = StrOutputParser()

# # --- Prompt Template ---
# template = """
# You are WYDOT chatbot, a polite and helpful Virtual Assistant of Wyoming Department of Transportation (WYDOT).
# Answer the question from the given context. Ensure clarity, brevity, and human-like responses.
# Context inside double backticks:``{context}```
# Question inside triple backticks:```{question}```
# If question is out of scope, answer it based on your role.
# Provide answers in a complete sentence concisely, within 50 words.
# JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.
# """
# prompt = ChatPromptTemplate.from_template(template)

# # --- Helper: fetch context text only ---
# def fetch_context(query: str):
#     docs = retriever.get_relevant_documents(query)
#     # Join top-k chunks into a single context block
#     combined_text = "\n\n".join(d.page_content for d in docs)
#     return combined_text, docs

# # --- Runnable chain ---
# chain = RunnableMap({
#     "context": lambda x: fetch_context(x["question"])[0],
#     "question": lambda x: x["question"]
# }) | prompt | llm | output_parser

# # --- Function to handle single query ---
# def resultDocuments(query: str):
#     result = chain.invoke({'question': query})
#     return {"text": result}

# # --- Async query processor for API calls ---
# async def query_processor(**request_params):
#     query = request_params.get('query', '')
#     if not query:
#         return {"result": "No query provided."}

#     try:
#         response = resultDocuments(query)
#         context_text, context_docs = fetch_context(query)
#         return {
#             "response": response, 
#             "context": [{"page": d.metadata.get("page"), "text": d.page_content[:300]} for d in context_docs]
#         }
#     except Exception as error:
#         print("Error processing query:", error)
#         return {"result": "Failed to process the query."}
