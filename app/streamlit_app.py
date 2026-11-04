"""Streamlit dashboard. Reads saved outputs from ``runs/<run_id>/``; it does not
run the pipeline itself, so the demo cannot fail on a live LLM call.

Planned pages: recommendations, drift by segment, stale vs refreshed retrieval,
drift vs quality impact, agent trace.
"""
