"""The only module that calls an LLM.

Provider-agnostic completion function (provider and models come from config,
keys from environment variables) with a disk cache keyed on provider, model,
prompt and parameters, so repeated runs cost nothing and are reproducible.
"""
