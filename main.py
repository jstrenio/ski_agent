from flask import Flask, render_template_string, request
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages, AnyMessage
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from rank_bm25 import BM25Okapi

import wikipedia
import pandas as pd
import ast
import re
import os
import io
import contextlib
import time

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

# =========================================================
# GEMINI
# =========================================================

models = [
    'gemini-3.1-flash-lite-preview',
    'models/gemma-4-31b-it',
    'gemini-2.5-flash-lite'
]

llm = ChatGoogleGenerativeAI(
    model=models[0],
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# =========================================================
# STATE
# =========================================================

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    plan: str
    action: str

# =========================================================
# LOAD BM25 DATABASE
# =========================================================

def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    return tokens + bigrams

df1 = pd.read_csv("ns_qa_p1_100.csv")
df2 = pd.read_csv("ns_qa_p1_200.csv")

forum_db = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['question'])

forum_db["answers"] = forum_db["answers"].apply(ast.literal_eval)
forum_db["answers"] = forum_db["answers"].apply(lambda msgs: ". ".join(msgs))

forum_db['thread'] = forum_db['question'] + '. ' + forum_db['answers']

tokenized_corpus = [tokenize(str(doc)) for doc in forum_db["thread"]]

bm25 = BM25Okapi(tokenized_corpus)

# =========================================================
# TOOLS
# =========================================================

@tool
def forum_search(query: str) -> str:
    """search ski forum discussions with provided query."""

    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    forum_db["score"] = scores

    results = forum_db.sort_values("score", ascending=False)

    discussions = results['thread'][:3].str[:2000].str.cat(sep='\n\n')

    print("\nFORUM SEARCH")
    print("=" * 60)
    print(discussions[:1000])

    return discussions


@tool
def wiki_search(query: str) -> str:
    """search wikipedia for top 5 most relevant pages."""

    try:
        results = wikipedia.search(query)[:5]

        print("\nWIKI SEARCH")
        print("=" * 60)
        print(results)

        return str(results)

    except Exception as e:
        print(f"wiki search failed: {e}")
        return "wiki search failed"


@tool
def wiki_summary(wiki_title: str) -> str:
    """retrieve wikipedia summary"""

    print("\nWIKI SUMMARY")
    print("=" * 60)
    print("input:", wiki_title)

    summary = "wikipedia unavailable"

    for _ in range(3):
        try:
            summary = wikipedia.page(
                wiki_title,
                auto_suggest=False
            ).content[:10000]

            break

        except:
            time.sleep(2)

    print(summary[:1000])

    return summary


llm_w_tools = llm.bind_tools([
    forum_search,
    wiki_search,
    wiki_summary
])

# =========================================================
# PROMPTS
# =========================================================

goal_prompt = ChatPromptTemplate.from_template("""
You are a planning agent for a ski forum database.

User Query:
{user_query}

Create a plan to answer the question.
""")

coordinate_prompt = ChatPromptTemplate.from_template("""
You are a task coordinating agent.

User query:
{messages}

Current plan:
{plan}

Decide next step:
- call tools if needed
- otherwise finish
""")

answer_prompt = ChatPromptTemplate.from_template("""
Use the retrieved information to answer thoroughly.

Context:
{messages}
""")

plan_chain = goal_prompt | llm
coordinate_chain = coordinate_prompt | llm_w_tools
answer_chain = answer_prompt | llm

# =========================================================
# GRAPH NODES
# =========================================================

def set_plan(state: State):

    user_query = state['messages'][-1].content

    response = plan_chain.invoke({
        'user_query': user_query
    })

    print("\nPLAN")
    print("=" * 60)
    print(response.content)

    return {
        'messages': [response],
        'plan': response.content
    }


def coordinate_action(state: State):

    response = coordinate_chain.invoke({
        'messages': state['messages'],
        'plan': state['plan']
    })

    print("\nCOORDINATOR")
    print("=" * 60)

    if response.tool_calls:
        print(response.tool_calls[0]["name"])
    else:
        print("END")

    return {
        'messages': [response]
    }


def route_action(state: State):

    last_msg = state['messages'][-1]

    tool_calls = (
        getattr(last_msg, "tool_calls", None)
        or last_msg.additional_kwargs.get("tool_calls")
    )

    if tool_calls:
        return "tools"

    return "END"


def answer_question(state: State):

    response = answer_chain.invoke({
        'messages': state['messages']
    })

    print("\nFINAL ANSWER")
    print("=" * 60)
    print(response.content)

    return {
        "messages": [response]
    }

# =========================================================
# BUILD GRAPH
# =========================================================

builder = StateGraph(State)

builder.add_node("set_plan", set_plan)
builder.add_node("coordinate_action", coordinate_action)
builder.add_node("answer_question", answer_question)

builder.add_node(
    "tools",
    ToolNode([
        forum_search,
        wiki_search,
        wiki_summary
    ])
)

builder.add_edge(START, "set_plan")

builder.add_edge(
    "set_plan",
    "coordinate_action"
)

builder.add_conditional_edges(
    "coordinate_action",
    route_action,
    {
        "tools": "tools",
        "END": "answer_question"
    }
)

builder.add_edge(
    "tools",
    "coordinate_action"
)

builder.add_edge(
    "answer_question",
    END
)

graph = builder.compile()

# =========================================================
# HTML
# =========================================================

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Ski Agent</title>

    <style>
        body {
            background: #111;
            color: #eee;
            font-family: Arial;
            padding: 30px;
        }

        textarea {
            width: 100%;
            height: 120px;
            background: #222;
            color: white;
            border: 1px solid #444;
            padding: 10px;
            font-size: 16px;
        }

        button {
            margin-top: 15px;
            padding: 12px 20px;
            font-size: 16px;
            cursor: pointer;
        }

        .output {
            margin-top: 30px;
            background: #1a1a1a;
            border: 1px solid #333;
            padding: 20px;
            height: 600px;
            overflow-y: scroll;
            white-space: pre-wrap;
        }
    </style>
</head>

<body>

    <h1>Ski Forum Agent</h1>

    <form method="POST">
        <textarea
            name="query"
            placeholder="Ask something..."
        >{{ query }}</textarea>

        <br>

        <button type="submit">
            Submit
        </button>
    </form>

    <div class="output">{{ output }}</div>

</body>
</html>
"""

# =========================================================
# ROUTE
# =========================================================

@app.route("/", methods=["GET", "POST"])
def home():

    output = ""
    query = ""

    if request.method == "POST":

        query = request.form.get("query")

        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):

            result = graph.invoke({
                "messages": [
                    HumanMessage(content=query)
                ]
            })

            final_answer = result['messages'][-1].content

            print("\n")
            print("=" * 60)
            print("FINAL OUTPUT")
            print("=" * 60)
            print(final_answer)

        output = buffer.getvalue()

    return render_template_string(
        HTML,
        output=output,
        query=query
    )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)