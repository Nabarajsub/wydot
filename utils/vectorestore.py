import os
from langchain.vectorstores.milvus import Milvus
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
host = os.getenv('HOST', 'localhost')
port = os.getenv('PORT', '19530')

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load PDF and split into documents
pdf_loader = PyPDFLoader('C:\\Users\\nsubedi1\\Desktop\\WYDOT project\\data\\Wyoming 2021 Standard Specifications for Road and Bridge Construction.pdf')
pdf_docs = pdf_loader.load()  # Load the documents from the PDF
# Create Milvus vector store from PDF documents
pdf_vectorstore = Milvus.from_documents(
    pdf_docs,
    embeddings,
    collection_name="wyodotspecs",
    connection_args={'host': host, 'port': port},
    vector_field="embedding"  # match your Milvus collection schema
)

print("PDF documents have been embedded and stored in Milvus.")


# # ------------------- CSV Example -------------------
# def load_csv_folder(folder_path: str):
#     csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
#     all_data = []
#     for csv_file in csv_files:
#         loader = CSVLoader(file_path=os.path.join(folder_path, csv_file), encoding="utf-8")
#         all_data.extend(loader.load())
#     return all_data

# csv_docs = load_csv_folder('C:\Users\ASUS\Desktop\INSURANCE\llm-banks\Chains\VectorStore\Data\GeneralBanking')

# csv_vectorstore = Milvus.from_documents(
#     csv_docs,
#     embeddings,
#     collection_name="muktinathGeneralInquiry",
#     connection_args={'host': host, 'port': port},
#     vector_field="embedding"
# )

# # ------------------- Word Document Example -------------------
# word_loader = UnstructuredWordDocumentLoader('C:\Users\ASUS\Desktop\INSURANCE\llm-banks\Chains\VectorStore\Data\BankDetails\aboutUs.docx')
# word_docs = word_loader.load()

# word_vectorstore = Milvus.from_documents(
#     word_docs,
#     embeddings,
#     collection_name="muktinathDetails",
#     connection_args={'host': host, 'port': port},
#     vector_field="embedding"
# )

# print("All documents have been embedded and stored in Milvus.")
