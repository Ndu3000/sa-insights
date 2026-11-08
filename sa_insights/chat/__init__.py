"""Retrieval and reasoning layer for the dashboard chatbot.

Three deliberately separate concerns:

``graph``
    A typed property graph over the Apache Ossie semantic model. Retrieval
    answers "what is relevant to this question"; it never answers with numbers.

``tools``
    Deterministic queries against the DuckDB warehouse. Every figure a user
    sees comes from here, so a number is either correct or the call failed —
    there is no third state where the model invents one.

``agent``
    The loop that decides which tools to call, asks for clarification when the
    question is genuinely ambiguous, and writes the prose around the results.
"""
