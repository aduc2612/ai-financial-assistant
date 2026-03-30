from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from typing import Literal, Annotated
from typing_extensions import TypedDict
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.tools import DuckDuckGoSearchResults

from dotenv import load_dotenv

from vector import retriever

SYSTEM_PROMPT = """You are a world-renowned export on financial matters.
            Focus only on facts and informations.
            Provide clear, concise answers based on logic and evidence.
            Always notify the user when stating personal opinions.
            Do not address emotions or provide emotional support.
            Be direct and straightforward in your responses.
            Explicitly state if there is not enough information to answer.
            """

load_dotenv()

model = ChatGroq(
    model="openai/gpt-oss-20b"
)

class MessageClassifier(BaseModel):
    message_type: Literal["historical", "current"] = Field(
        ...,
        description = "Classify if the message requires looking at historical data or searching for up-to-date data"
    )

class State(TypedDict):
    messages: Annotated[list, add_messages]
    message_type: str | None

def classify_message(state: State):
    last_message = state["messages"][-1]
    classifier_model = model.with_structured_output(MessageClassifier)

    result = classifier_model.invoke([
        {
            "role": "system",
            "content": """Classify the user message as either:
            - 'historical': If it asks for past trends, historical datas
            - 'current': If it explicitly asks for current events, trends, datas
            """
        },
        {"role": "user", "content": last_message.content}
    ])
    return {"message_type": result.message_type}

def router(state: State):
    message_type = state.get("message_type", "historical")
    if message_type == "historical":
        return {"next": "historical"}

    return {"next": "current"}

def use_historical_data(state: State):
    last_message = state["messages"][-1]

    reports = retriever.invoke(last_message.content)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""Here are some reports of historical market events: {reports}
                        {last_message.content}
"""
        }
    ]
    reply = model.invoke(messages)
    return {"messages": [{"role": "assistant", "content": f"Use historical:\n{reply.content}"}]}

def use_current_data(state: State):
    last_message = state["messages"][-1]

    wrapper = DuckDuckGoSearchAPIWrapper(max_results=5)
    search = DuckDuckGoSearchResults(api_wrapper=wrapper, source="news")

    # Run a query
    news = search.run(last_message.content)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"""Here are some current market news: {news}
                        {last_message.content}
"""
        }
    ]
    reply = model.invoke(messages)
    return {"messages": [{"role": "assistant", "content": f"Use current:\n{reply.content}"}]}

graph_builder = StateGraph(State)
graph_builder.add_node("classifier", classify_message)
graph_builder.add_node("router", router)
graph_builder.add_node("historical", use_historical_data)
graph_builder.add_node("current", use_current_data)

graph_builder.add_edge(START, "classifier")
graph_builder.add_edge("classifier", "router")

graph_builder.add_conditional_edges(
    "router",
    lambda state: state.get("next"), 
    {"historical": "historical", "current": "current"}
)

graph_builder.add_edge("historical", END)
graph_builder.add_edge("current", END)

graph = graph_builder.compile()


def run_chatbot():
    state = {"messages": [], "message_type": None}

    while True:
        user_input = input("Ask me a question (q to exit): ")
        if user_input == "q":
            print("Bye")
            break

        state["messages"] = state.get("messages", []) + [
            {"role": "user", "content": user_input}
        ]

        state = graph.invoke(state)

        if state.get("messages") and len(state["messages"]) > 0:
            last_message = state["messages"][-1]
            print(f"Assistant: {last_message.content}")


if __name__ == "__main__":
    run_chatbot()