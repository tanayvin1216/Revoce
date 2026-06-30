"""Phase 3 — autonomous self-test harness + the coverage-shaped metric reframe.

Everything here is ADDITIVE. The seam (rank_vocabulary.py) is never touched: the
harness drives it through its existing injectable `ranker` and `strategy`/`topics`
arguments, exactly as the server does.
"""
