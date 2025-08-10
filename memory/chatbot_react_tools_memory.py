from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage, AIMessage
from langgraph.graph import START, END, StateGraph, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated
from langchain_tavily import TavilySearch
from langchain.agents.react.agent import create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
from langchain import hub
import sqlite3

load_dotenv()
# Memory database setup
checkpoint_db = "chatbot_memory.db"
conn = sqlite3.connect(checkpoint_db, check_same_thread = False)
memory = SqliteSaver(conn)

# Define the state
class ChatState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]

llm = ChatGroq(model="llama3-70b-8192")

# Define a search tool
search_tool = TavilySearch()
from langchain_core.tools import tool

# Define a search tool
search_tool = TavilySearch()

@tool
def calculator(expression: str) -> str:
    """
    Perform basic math calculations. 
    Input: A simple math expression like "2+2", "5*3", "10-4", "8/2"
    """
    try:
        # Remove spaces and validate the expression
        expression = expression.strip().replace(" ", "")
        
        # Basic validation - only allow numbers and basic operators
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return "Error: Only basic math operations (+, -, *, /) and numbers are allowed"
        
        # Evaluate the expression safely
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"

tools = [search_tool, calculator]

# Much clearer prompt template
react_template = """You are a helpful AI assistant that uses tools to answer questions.

Available tools:
{tools}

ALWAYS follow this exact format:

Question: the input question
Thought: think about what you need to do
Action: choose from [{tool_names}]
Action Input: the exact input for the tool
Observation: the result from the tool
... (repeat Thought/Action/Action Input/Observation as needed)
Thought: now I have enough information to answer
Final Answer: your complete answer to the question

EXAMPLES:

For math: 
Question: What is 2+2?
Thought: I need to calculate 2+2
Action: calculator
Action Input: 2+2
Observation: 2+2 = 4
Thought: now I have the answer
Final Answer: 2+2 equals 4.

For search:
Question: What's the weather in Paris?
Thought: I need to search for current weather in Paris
Action: tavily_search
Action Input: current weather Paris today
Observation: [search results with weather data]
Thought: I found the weather information
Final Answer: Based on current data, the weather in Paris is...

IMPORTANT: 
- Always provide a Final Answer
- Never use "Action: None"
- Analyze search results in your Thought before giving Final Answer

Question: {input}
Thought: {agent_scratchpad}"""

prompt = PromptTemplate(
    input_variables=["input", "tools", "tool_names", "agent_scratchpad"],
    template=react_template
)

# Create the agent
react_agent = create_react_agent(
    llm=llm, 
    tools=tools, 
    prompt=prompt
)

# Create the executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=react_agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=6,
    max_execution_time=60,
    return_intermediate_steps=True,
    early_stopping_method="generate"
)

graph = StateGraph(ChatState)

# Defining the nodes
def chat_node(state: ChatState) -> ChatState:
    response = agent_executor.invoke({"input": state.messages})
    return {"messages": AIMessage(content=response['output'])}
    
# Add the chat node to the graph
graph.add_node(chat_node, "chat_node")

# Add edges to the graph
graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

app = graph.compile(checkpointer=memory)
config = {"configurable":{
    "thread_id": "2_react_agent"
}}

# Chatbot start
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot.")
        break
    
    response = app.invoke({"messages": HumanMessage(user_input)}, config=config)
    
    print(f"AI: {response['messages'][-1].content}")