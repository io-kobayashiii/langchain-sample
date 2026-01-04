from typing import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage


class State(TypedDict):
    # messagesの型は`list`である。
    # アノテーションに指定した`add_messages`関数はこのステートの更新方法を定義している。
    # この例では既存のメッセージを置き換えるのではなくリストへ追記する。
    messages: Annotated[list, add_messages]


builder = StateGraph(State)

model = ChatOpenAI(model="gpt-4o-mini")


def chatbot(state: State):
    answer = model.invoke(state["messages"])
    return {"messages": [answer]}


# 第1引数はノードの一意の名前
# 第2引数は実行する関数または実行可能（Runnable）オブジェクト
builder.add_node("chatbot", chatbot)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

graph = builder.compile()
user_input = {"messages": [HumanMessage("hi!")]}
for chunk in graph.stream(user_input):
    print(chunk)
