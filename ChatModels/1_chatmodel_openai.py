from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

chat_model = ChatOpenAI(model='gpt-4.1', temperatur=0.3, max_completion_tokens=10)

ceo = chat_model.invoke('Who is the CEO of Open AI')

print(ceo)