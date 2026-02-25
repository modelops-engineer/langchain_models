from langchain_openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

openai_llm = OpenAI(model='gpt-3.5-turbo-instruct')

ceo = openai_llm.invoke('Who is the CEO of OpenAI')

print(ceo)