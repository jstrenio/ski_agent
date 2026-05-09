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
import csv
import ast
import re
import os
import io
import contextlib
import time
import sys

csv.field_size_limit(sys.maxsize)

# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

# =========================================================
# GEMINI
# =========================================================

models = ['gemini-3.1-flash-lite-preview']

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
# TOKENIZER
# =========================================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())

# =========================================================
# LOAD CSV
# =========================================================

threads = []
tokenized_corpus = []
seen_questions = set()

CSV_FILES = [
    "ns_qa_p1_100.csv",
    "ns_qa_p1_200.csv"
]

for filename in CSV_FILES:

    with open(filename, newline="", encoding="utf-8") as f:

        reader = csv.DictReader(f)

        for row in reader:

            question = (row.get("question") or "").strip()

            if not question:
                continue

            if question in seen_questions:
                continue

            seen_questions.add(question)

            try:
                answers = ast.literal_eval(
                    row.get("answers", "[]")
                )

                if isinstance(answers, list):
                    answers = ". ".join(answers)

            except:
                answers = ""

            thread = f"{question}. {answers}"

            threads.append(thread)

            tokenized_corpus.append(
                tokenize(thread)
            )

print(f"loaded threads: {len(threads)}")

# =========================================================
# BM25
# =========================================================

bm25 = BM25Okapi(tokenized_corpus)

print("bm25 ready")

# =========================================================
# TOOLS
# =========================================================

@tool
def forum_search(query: str) -> str:
    """search ski forum discussions with provided query."""

    tokenized_query = tokenize(query)

    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:3]

    selected_threads = [
        threads[i][:2000]
        for i in top_indices
    ]

    discussions = "\n\n".join(selected_threads)

    print("forum results:")

    for i, thread in enumerate(selected_threads, 1):

        print(f"\n{'=' * 30} THREAD {i} {'=' * 30}\n")

        print(thread[:350])

    print('\n' + '=' * 60 + '\n')

    return discussions

@tool
def wiki_search(query: str) -> str:
    """search wikipedia for top 5 most relevant pages."""

    try:

        results = wikipedia.search(query)[:5]

        print('wiki search results:' + str(results))

        return str(results)

    except Exception as e:

        print(f'wiki search failed: {e}')

        return "[]"

@tool
def wiki_summary(wiki_title: str) -> str:
    """takes title from wiki_search and retrieves summary."""

    print("started wiki_summary...")
    print("input:", wiki_title)

    summary = 'wikipedia unavailable'

    for _ in range(3):

        try:

            summary = wikipedia.page(
                wiki_title,
                auto_suggest=False
            ).content[:10000]

            break

        except:

            print('summary failed')

            time.sleep(2)

            try:

                summary = wikipedia.summary(
                    wiki_title,
                    auto_suggest=False
                )[:10000]

                break

            except:

                print('shortened summary also failed.')

    print('wiki summary: ' + str(summary[:100]))

    print('\n' + '=' * 60 + '\n')

    return summary

# =========================================================
# TOOL LLM
# =========================================================

llm_w_tools = llm.bind_tools([
    forum_search,
    wiki_search,
    wiki_summary
])

# =========================================================
# PROMPTS
# =========================================================

goal_text = '''
You are a planning agent at a ski forum database of public conversations on all topics ski related. Your job is to interpret the user's query
and devise a course of action to answer the question. Your options are to use the forum_search tool to search forum conversations, to use the
wiki_search tool followed by the wiki_summary tool to search wikipedia and extract information, or to answer on your own. When possible you should consider using one or both tools as many times as needed to retrieve the necessary information to answer the query.
User Query:{user_query}
'''

goal_prompt = ChatPromptTemplate.from_template(goal_text)

coordinate_text = '''
You are a task coordinating agent.

User query:
{messages}

Current plan:
{plan}

Previous tool results are included above.
Decide the next step:
- If more info is needed, call a tool
- If done, respond with END
'''

coordinate_prompt = ChatPromptTemplate.from_template(coordinate_text)

answer_text = '''
Use the retrieved information to write a thorough answer to the user's query, rooted in the information collected. Context: {messages}
'''

answer_prompt = ChatPromptTemplate.from_template(answer_text)

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

    print('ai plan: ' + response.content[0]['text'])

    print('\n' + '=' * 60 + '\n')

    return {
        'messages': [response],
        'plan': response.content
    }

def coordinate_action(state: State):

    response = coordinate_chain.invoke({
        'messages': state['messages'],
        'plan': state['plan']
    })

    print(
        f'coordinated action: '
        f'{response.tool_calls[:1][0]["name"]} '
        f'{response.tool_calls[:1][0]["args"].get("query", "")}'
        if response.tool_calls else "END"
    )

    print('\n' + '=' * 60 + '\n')

    return {'messages': [response]}

def route_action(state: State):

    last_msg = state['messages'][-1]

    print('routing tools call...')

    tool_calls = (
        getattr(last_msg, "tool_calls", None)
        or last_msg.additional_kwargs.get("tool_calls")
    )

    if tool_calls:
        return 'tools'

    return 'END'

def answer_question(state: State):

    response = answer_chain.invoke({
        'messages': state['messages']
    })

    return {"messages": [response]}

# =========================================================
# BUILD GRAPH
# =========================================================

builder = StateGraph(State)

builder.add_node('set_plan', set_plan)

builder.add_node(
    'coordinate_action',
    coordinate_action
)

builder.add_node(
    'answer_question',
    answer_question
)

builder.add_node(
    'tools',
    ToolNode([
        forum_search,
        wiki_search,
        wiki_summary
    ])
)

builder.add_edge(
    START,
    'set_plan'
)

builder.add_edge(
    'set_plan',
    'coordinate_action'
)

builder.add_conditional_edges(
    'coordinate_action',
    route_action,
    {
        'tools': 'tools',
        'END': 'answer_question'
    }
)

builder.add_edge(
    'tools',
    'coordinate_action'
)

builder.add_edge(
    'answer_question',
    END
)

graph = builder.compile()

print("graph compiled")

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
    height: 650px;
    overflow-y: scroll;
    white-space: pre-wrap;
    line-height: 1.5;
}

</style>

</head>

<body>

<h1>Ski Forum Agent</h1>

<form method="POST">

<textarea name="query">how does public opinion of the XGames differ from the actual business or event?</textarea>

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

    if request.method == "POST":

        query = request.form.get("query", "")

        buffer = io.StringIO()

        with contextlib.redirect_stdout(buffer):

            try:

                result = graph.invoke({
                    "messages": [
                        HumanMessage(content=query)
                    ]
                })

                ans = result['messages'][-1].content[0]['text']

                print('\n' + '=' * 60 + '\n')

                print(ans)

            except Exception as e:

                print("\nERROR")
                print("=" * 60)
                print(str(e))

        output = buffer.getvalue()

    return render_template_string(
        HTML,
        output=output
    )

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    port = int(os.environ.get("PORT", 5000))

    app.run(
        host="0.0.0.0",
        port=port
    )