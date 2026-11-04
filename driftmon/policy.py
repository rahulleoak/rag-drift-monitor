"""Decision rules: drift and impact evidence to corrective actions.

Plain, table-tested rules with no LLM involvement, for example:
    drift + impact, refreshed index recovers quality -> re-index / refresh cadence
    drift + impact, refreshed index still poor       -> evaluate embedding model
    retrieval fine, answers poor or ungrounded       -> revise prompts
    drift, no material impact                        -> continue monitoring
    impact without drift, or too little evidence     -> escalate to a human
Every action must cite the evidence ids it was derived from.
"""
