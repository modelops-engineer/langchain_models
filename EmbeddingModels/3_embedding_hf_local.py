from langchain-huggingface import HuggingFaceEmbeddings as hfe

docs = [
    "Who is the CEO of Open AI?",
    "Embedding model",
    "Embedding of a document"
]

model = hfe(model_name = 'sentence-transformer/all-MiniLM-L6-v2')

embedding = model.embed_query('Who is the CEO of Open AI?')
vector = model.embed_documentss(docs)

print(str(embedding))