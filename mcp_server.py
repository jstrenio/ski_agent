from mcp.server.fastmcp import FastMCP
import pandas as pd
import ast
from rank_bm25 import BM25Okapi
import re

# load csv
df1 = pd.read_csv("ns_qa_p1_100.csv")
df2 = pd.read_csv("ns_qa_p1_200.csv")

forum_db = pd.concat([df1, df2], ignore_index=True).drop_duplicates(subset=['question'])

# convert stringified list -> actual list
forum_db["answers"] = forum_db["answers"].apply(ast.literal_eval)

# optional: combine messages into one string per row
forum_db["answers"] = forum_db["answers"].apply(lambda msgs: ". ".join(msgs))

forum_db['answers'].fillna('')
forum_db['question'].fillna('')
forum_db['thread'] = forum_db['question'] + '. ' + forum_db['answers']

def tokenize(text):
    tokens = re.findall(r"\b\w+\b", text.lower())
    
    # add bigrams
    bigrams = [" ".join(tokens[i:i+2]) for i in range(len(tokens)-1)]
    
    return tokens + bigrams

# tokenize corpus
tokenized_corpus = [tokenize(str(doc)) for doc in forum_db["thread"]]

# build BM25 index
bm25 = BM25Okapi(tokenized_corpus)

mcp = FastMCP("forum_mcp")

@mcp.tool()
def forum_search(query: str) -> str:
    """search ski forum discussions with provided query."""
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    forum_db["score"] = scores
    results = forum_db.sort_values("score", ascending=False)
    discussions = results['thread'][:5].str[:1000].str.cat(sep='\n\n')
    print('forum findings: ' + discussions[:100])

    return discussions

if __name__ == "__main__":
    # This hosts the server on http://localhost:8000/sse by default
    mcp.run(transport="sse")


# from mcp.server.fastmcp import FastMCP

# mcp = FastMCP("forum_mcp")

# @mcp.tool()
# def add_one(value: int) -> int:
#     return value + 1


