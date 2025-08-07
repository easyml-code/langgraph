from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(model="llama3-70b-8192")

response = llm.invoke("What is the capital of France?")

print(response)