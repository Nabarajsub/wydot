# import os
# from langchain.vectorstores.milvus import Milvus
# from langchain.document_loaders import PyPDFLoader
# from langchain.embeddings import HuggingFaceEmbeddings
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()
# host = os.getenv('HOST', 'localhost')
# port = os.getenv('PORT', '19530')

# # Initialize embeddings
# embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# # Load PDF and split into documents
# pdf_loader = PyPDFLoader('C:\\Users\\nsubedi1\\Desktop\\WYDOT project\\data\\Wyoming 2021 Standard Specifications for Road and Bridge Construction.pdf')
# pdf_docs = pdf_loader.load()  # Load the documents from the PDF
# # Create Milvus vector store from PDF documents
# pdf_vectorstore = Milvus.from_documents(
#     pdf_docs,
#     embeddings,
#     collection_name="wyodotspecs",
#     connection_args={'host': host, 'port': port},
#     vector_field="embedding"  # match your Milvus collection schema
# )

# print("PDF documents have been embedded and stored in Milvus.")
import os
from dotenv import load_dotenv

# LangChain (Milvus vector store)
try:
    from langchain.vectorstores.milvus import Milvus
except Exception:
    # if your langchain version moved Milvus to community
    from langchain_community.vectorstores import Milvus

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

# --- Zilliz Cloud connection (TLS) ---
URI = os.getenv("MILVUS_URI")  # e.g. xxxxx.api.gcp-us-west1.zillizcloud.com:19530
TOKEN = os.getenv("MILVUS_TOKEN")
SECURE = os.getenv("MILVUS_SECURE", "false").lower() in ("1", "true", "yes")
COLLECTION = os.getenv("MILVUS_COLLECTION", "wyodotspecs_cloud")

conn_args = {
    # prepend https:// for TLS endpoints
    "uri": f"https://{URI}" if URI and not URI.startswith("http") else (URI or ""),
    "token": TOKEN,
    "secure": SECURE,
}

# --- Embeddings (384-dim) ---
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},  # cosine ≈ inner product with normalized vectors
)

# --- Load & chunk PDF ---
pdf_path = r"C:\Users\nsubedi1\Desktop\WYDOT project\data\Wyoming 2021 Standard Specifications for Road and Bridge Construction.pdf"
loader = PyPDFLoader(pdf_path)
pages = loader.load()

# chunking is important for better recall + cheaper queries
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
docs = splitter.split_documents(pages)

# --- Create / upsert into Zilliz collection ---
# AUTOINDEX picks a good index; metric_type "IP" matches normalized embeddings.
pdf_vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name=COLLECTION,
    connection_args=conn_args,
    text_field="text",
    vector_field="embedding",
    index_params={"index_type": "AUTOINDEX", "metric_type": "IP"},
)

print(f"✅ {len(docs)} chunks embedded and stored in Zilliz collection: {COLLECTION}")


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
