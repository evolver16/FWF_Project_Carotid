"""python -m studies <study> [positional ...] [key=value ...], e.g. python -m studies pressure_step p_gr=0.012 T=50"""

import argparse
import ast

from studies import STUDIES
from studies.common import RESULTS


def value(v):
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


ap = argparse.ArgumentParser(prog="python -m studies")
ap.add_argument("study", choices=list(STUDIES))
ap.add_argument("args", nargs="*", help="positional, or key=value for keyword arguments (pressure_step: any artery_par key)")
a = ap.parse_args()
RESULTS.mkdir(parents=True, exist_ok=True)
STUDIES[a.study](*[v for v in a.args if "=" not in v],
                 **{k: value(v) for k, v in (s.split("=", 1) for s in a.args if "=" in s)})
