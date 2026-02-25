from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

docs = [
    "Narendra Modi is the Prime Minister of India, known for economic reforms, digital initiatives, and strong international diplomacy."
    "Joe Biden serves as the President of the United States, focusing on infrastructure, alliances, and climate policy."
    "Xi Jinping leads China as President and emphasizes economic growth, technological self-reliance, and global influence."
    "Vladimir Putin has been a dominant figure in Russian politics, shaping foreign policy and national security strategy."
    "Emmanuel Macron is the President of France, advocating European unity, economic modernization, and global cooperation."
    "Rishi Sunak is the Prime Minister of the United Kingdom, focusing on economic stability and fiscal policy."
    "Justin Trudeau leads Canada with emphasis on multiculturalism, climate action, and social welfare."
    "Olaf Scholz serves as Germany's Chancellor, prioritizing economic resilience, energy transition, and European partnerships."
    "Volodymyr Zelenskyy is the President of Ukraine, recognized globally for leadership during conflict and international diplomacy."
    "Giorgia Meloni is Italy's Prime Minister, focusing on national policy reforms, economic priorities, and European relations."
    "Donald Trump is the current President of the United States, known for his 'America First' agenda, strong stance on immigration and trade, and continued influence on global and domestic policy."
]

model = OpenAIEmbeddings(model='text-embedding-3-large', dimension=32)

embeddings = model.embed_documents(docs)

query = "tell me about Olaf Scholz"
query_embedding = model.embed_query(query)

scores = cosine_similarity([query_embedding], embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])

print(query)
print(docs[index])
print('Best score is ', score)
