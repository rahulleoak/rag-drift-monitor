"""driftmon: agentic semantic drift detection and RAG quality assurance.

Layout rule: agents in ``driftmon.agents`` are thin. They read the shared
state, call library code, and write one typed report. Statistics, metrics and
decisions live in plain modules (``drift``, ``evaluation``, ``policy``) that
never import ``driftmon.llm``. Only ``llm.py`` talks to the LLM.
"""
