from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='')

result = model.invoke('Who is the powerful leader in the world?')

print(result.content)
