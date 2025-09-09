from utils.embeddings import get_llm_embedings
from langchain.vectorstores.milvus import Milvus 
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableMap
from langchain.schema.output_parser import StrOutputParser
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.tools import Tool

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
load_dotenv()
host = os.getenv('HOST')
port = os.getenv('PORT')
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize LLM with Llama-3-70B-8192
llm = ChatGroq(
    model="llama-3-70b-8192",
    temperature=0,
    api_key=groq_api_key
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

database = Milvus(embeddings,connection_args={'host':host,'port':port},collection_name='wyodotspecs')
output_parser = StrOutputParser()

# Define the prompt template
template = """
You are WYDOT chatbot, a polite and helpful Virtual Assistant of Wyoming Department of Transportation (WYDOT).
Answer the question from the given context. Ensure clarity, brevity, and human-like responses.
Context inside double backticks:``{context}```
Question inside triple backticks:```{question}```
If question is out of scope, answer it based on your role.
Provide answers in a complete sentence concisely, within 50 words.
JUST PROVIDE THE ANSWER IN ENGLISH WITHOUT ``` AND NOTHING ELSE.
"""

# Create the prompt template
prompt = ChatPromptTemplate.from_template(template)

# Create the chain
chain = RunnableMap({
    "context": lambda x: database.similarity_search(x['question'], k=5),
    "question": lambda x: x['question']
}) | prompt | llm | output_parser

# Define the function to handle queries
def resultDocuments(query: str):
    result = chain.invoke({'question': query})
    print(result)
    return {"text": result}

# Create the tool instance
result_documents_tool = Tool(
    name="result_documents_tool",
    func=resultDocuments,
    description="A tool that provides answers related to documents and queries related to Wyoming Department of Transportation (WYDOT) specifications and technical documents."
)

# Example usage
# result = result_documents_tool.func("What are the required documents for opening an account?")
# print(result)
