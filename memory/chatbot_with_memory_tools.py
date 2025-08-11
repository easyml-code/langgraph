from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_tavily import TavilySearch
from langchain_core.tools import Tool
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated
from langgraph.prebuilt import ToolNode
import sqlite3

load_dotenv()
# Memory database setup
checkpoint_db = "chatbot_memory.db"
conn = sqlite3.connect(checkpoint_db, check_same_thread=False)
memory = SqliteSaver(conn)


class ChatState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]


search_tool = TavilySearch(max_results=2)

tavily_search = Tool(
    name="Search Tool", 
    func=search_tool, 
    description="A tool to search the web for information."
    )
tools = [tavily_search]

# llm = ChatGroq(model="openai/gpt-oss-20b")
llm = ChatGroq(model="llama3-70b-8192")

llm_with_tools = llm.bind_tools(tools)

def chat_node(state: ChatState) -> ChatState:
    response = llm_with_tools.invoke(state.messages)
    return {"messages": response}

def router_node(state: ChatState):
    last_message = state.messages[-1]

    if(hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0):
        return "tool_node"
    else: 
        return END
    
tool_node = ToolNode(tools=tools)

# Create the state graph
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tool_node", tool_node)

# Add edges to the graph
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", router_node)
graph.add_edge("tool_node", "chat_node")

app = graph.compile(checkpointer=memory)

config = {"configurable":{
    "thread_id": "tool_3"
}}

# Chatbot start
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot.")
        break
    
    response = app.invoke({"messages": HumanMessage(user_input)}, config=config)
    
    print(f"AI: {response['messages'][-1].content}")