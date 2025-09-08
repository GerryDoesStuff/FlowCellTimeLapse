from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd

# Placeholder for per-frame parallelization hooks.
# The current pipeline registers each frame against the reference sequentially
# because ECC depends on a shared overlap mask; however, heavy segmentation and
# I/O can still be parallelized in future iterations.

def cpu_count() -> int:
    try:
        return mp.cpu_count()
    except NotImplementedError:
        return 4
