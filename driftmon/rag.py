"""The RAG pipeline under test: retrieve top-k, then generate an answer.

Takes an index (stale or refreshed) and a question, returns the retrieved
documents and the generated answer. Used by Agents C and D.
"""
