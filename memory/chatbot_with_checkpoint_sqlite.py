from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated
import sqlite3

load_dotenv()
# Memory database setup
checkpoint_db = "chatbot_memory.db"
conn = sqlite3.connect(checkpoint_db, check_same_thread = False)
memory = SqliteSaver(conn)

# Define the state
class ChatState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGroq(model="openai/gpt-oss-20b")

# Define a simple state graph
graph = StateGraph(ChatState)

# Defining the nodes
def chat_node(state: ChatState) -> ChatState:
    response = llm.invoke(state.messages)
    return {"messages": AIMessage(content=response.content)}
    
# Add the chat node to the graph
graph.add_node(chat_node, "chat_node")

# Add edges to the graph
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

app = graph.compile(checkpointer=memory)

conversation_state = ChatState(messages=[
    SystemMessage(content="You are a helpful assistant.")
])

config = {"configurable":{
    "thread_id": "1"
}}

# Chatbot start
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot.")
        break
    
    response = app.invoke({"messages": HumanMessage(user_input)}, config=config)
    
    print(f"AI: {response['messages'][-1].content}")