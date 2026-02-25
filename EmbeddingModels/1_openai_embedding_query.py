from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

model = OpenAIEmbeddings(model='text-embedding-3-lasrge', dimensions=32)

embedding = model.embed_query('Who is the CEO of Open AI?')

print(str(embedding))