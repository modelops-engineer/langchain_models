from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    "Who is the CEO of Open AI?",
    "Embedding model",
    "Embedding of a document"
]

model = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

embedding = model.embed_query('docs')

print(str(embedding))