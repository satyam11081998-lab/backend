"""MECE Prep Copilot v2 - isolated, role/company-aware prep stack.

Fully separate from the live case/guesstimate interview + scoring pipeline:
its own engine copy, its own GPT scorer copy, its own corpus/research, its own
route and tables. Nothing here is imported by the live pipeline, and nothing
here imports-to-mutate the originals. Flag-gated by COPILOT_V2_ENABLED.
"""
