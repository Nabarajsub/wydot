import os
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()


from langchain_openai import OpenAIEmbeddings, ChatOpenAI

OPEN_API_KEY= os.getenv('OPENAI_API_KEY')

llm_imbeding_dict={
    "open_ai":{
        "llm":ChatOpenAI(openai_api_key=OPEN_API_KEY,temperature=0,model="gpt-3.5-turbo"),
        "embedding":OpenAIEmbeddings(),
        "embedings_1" : HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1", encode_kwargs = {'normalize_embeddings': True}),
        },
    
    "gemini":{},
    "hugging_face":{}
}


#embeding type must be [ "open_ai","gemini" ]
def get_llm_embedings(embeding_type:str):
    return llm_imbeding_dict.get(embeding_type,None)