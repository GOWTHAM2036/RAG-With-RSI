import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from typing import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph import START, END

load_dotenv()

model = init_chat_model(
    model="gemini-2.5-flash",
    model_provider="google_genai",
    temperature=0.3,
    max_tokens=512,
    max_retries=3,
    api_key = os.getenv("gemini_api_key")
)

# 2. Define the state
class State(TypedDict):
    question: str
    answer: str


# 3. Create the chatbot node
def chatbot(state):
    response = model.invoke(state["question"])

    return {
        "answer": response.content
    }


# 4. Build the graph
builder = StateGraph(State)

builder.add_node("chatbot", chatbot)

builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)


# 5. Compile the graph
graph = builder.compile()


# 6. Run the graph
result = graph.invoke({
    "question": "What is LangGraph?"
})


# 7. Print the final state
print(result)