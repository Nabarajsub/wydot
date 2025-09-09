import nest_asyncio
from dotenv import load_dotenv
from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
load_dotenv()
nest_asyncio.apply()

from llama_parse import LlamaParse

instructions = """
        Extract information from the given pdf in an arranged manner.
        """
parser = LlamaParse(api_key=os.getenv("llamaparseapi"), result_type="markdown", parsing_instruction=instructions)
documents = parser.load_data("C:\\Users\\ASUS\\Desktop\\INSURANCE\\llm-banks\\Chains\\VectorStore\\Data\\PdfData\\InterestRates.docx")
print(len(documents))
# print(documents.text)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=1000,
    chunk_overlap=100   
)
texts = text_splitter.create_documents([documents[0].text])
updated_pages=[]
for text in texts:
    updated_pages.append(text)

print(updated_pages)