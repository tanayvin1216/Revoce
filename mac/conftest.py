"""Pytest bootstrap: put mac/ on sys.path so the seam modules import the same
way they do at runtime (`from rank_vocabulary import ...`), regardless of the
directory pytest is launched from.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
