import os
import sqlite3
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

connection = sqlite3.connect("history.db")
cursor = connection.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS messages (
    session_id TEXT,
    role TEXT,
    message TEXT
)
""")

connection.commit()
connection.close()

load_dotenv()

llm = init_chat_model(
    "google_genai:gemini-2.5-flash",
    temperature=0.3,
    max_tokens=1024,
    max_retries=3,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm | StrOutputParser()

def save_message(session_id, role, message):
    connection = sqlite3.connect("history.db")
    cursor = connection.cursor()
    cursor.execute(
        """
        INSERT INTO messages(session_id, role, message)
        VALUES (?, ?, ?)
        """,
        (session_id, role, message)
    )
    connection.commit()
    connection.close()

def load_history(session_id):
    connection = sqlite3.connect("history.db")
    cursor = connection.cursor()
    cursor.execute(
        """
        SELECT role, message
        FROM messages
        WHERE session_id = ?
        """,
        (session_id,)
    )
    rows = cursor.fetchall()
    connection.close()
    history = []
    for role, message in rows:
        if role == "human":
            history.append(HumanMessage(content=message))
        elif role == "ai":
            history.append(AIMessage(content=message))
    return history


session_id = "user_1"

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Quitting...")
        break

    history = load_history(session_id)
    response = chain.invoke({
        "input": user_input,
        "history": history
    })
    save_message(session_id, "human", user_input)
    save_message(session_id, "ai", response)

    print(f"AI: {response}")