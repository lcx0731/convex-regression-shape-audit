"""Small exact rational reference by enumerating every constraint face.

This intentionally does not call the production matrix builder or solver.
At most nine distinct sorted observations are supported.
"""

from fractions import Fraction
from itertools import combinations


def _solve(matrix, rhs):
    n = len(rhs)
    rows = [list(row)+[b] for row, b in zip(matrix, rhs)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        divisor = rows[j][j]
        rows[j] = [v/divisor for v in rows[j]]
        for i in range(n):
            if i != j:
                coefficient = rows[i][j]
                rows[i] = [u-coefficient*v for u, v in zip(rows[i], rows[j])]
    return [row[-1] for row in rows]


def exact_projection(x, y, weights=None):
    """Return exact fitted values and SSE; float inputs mean exact binary values."""
    x, y = list(map(Fraction, x)), list(map(Fraction, y))
    weights = [Fraction(1)]*len(x) if weights is None else list(map(Fraction, weights))
    n = len(x)
    if not 2 <= n <= 9 or len(y) != n or len(weights) != n or any(w <= 0 for w in weights) or any(a >= b for a, b in zip(x, x[1:])):
        raise ValueError("reference needs 2 to 9 sorted distinct observations with positive weights")
    rows = []
    for i in range(1, n-1):
        row = [Fraction(0)]*n
        left, right = x[i]-x[i-1], x[i+1]-x[i]
        row[i-1], row[i], row[i+1] = 1/left, -1/left-1/right, 1/right
        rows.append(row)
    dot = lambda a, b: sum(u*v for u, v in zip(a, b))
    best, count = None, 0
    for size in range(n-1):
        for active in combinations(range(n-2), size):
            count += 1
            selected = [rows[j] for j in active]
            gram = [[sum(a[k]*b[k]/weights[k] for k in range(n)) for b in selected] for a in selected]
            multipliers = _solve(gram, [-dot(row, y) for row in selected])
            candidate = [y[k]+sum(q*row[k] for q, row in zip(multipliers, selected))/weights[k] for k in range(n)]
            if any(dot(row, candidate) < 0 for row in rows):
                continue
            sse = sum(w*(a-b)**2 for w, a, b in zip(weights, candidate, y))
            if best is None or sse < best[0]:
                best = (sse, candidate, active)
    if best is None:
        raise ArithmeticError("no feasible face found")
    return {"fitted": best[1], "weighted_sse": best[0], "active_equalities": best[2], "faces_examined": count}
