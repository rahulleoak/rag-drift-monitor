"""driftmon: agentic semantic drift detection and RAG quality assurance.

Layout: agents in ``driftmon.agents`` are thin. They read the shared state,
call library code, and write one typed report (``schemas.py``); ``graph.py``
wires them together. Statistics, metrics and decisions live in plain, tested
code (``drift``, ``evaluation``, ``policy.py``). Every LLM call goes through
``llm.client``, which caches, budgets and logs it.

Sub-packages: ``data`` (what comes in), ``rag`` (the pipeline under test),
``llm``, ``drift``, ``evaluation``, ``longform`` (Wikipedia and chunking) and
``dashboard`` (charts and run loading).
"""
