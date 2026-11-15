"""Operational scripts: exporters, evaluation, and deployment.

A package marker so `scripts.eval_chat` can be imported by the scheduled Modal
evaluation as well as run from the command line. Without it the nightly job and
the deploy gate would each need their own copy of the case list, and the two
would drift.
"""
