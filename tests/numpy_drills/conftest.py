import ast
import inspect
import textwrap

import pytest

_LOOPS = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
_DISGUISED_LOOPS = {"vectorize", "apply_along_axis", "apply_over_axes", "frompyfunc", "fromiter", "map"}


def _check(fn, banned):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and "NotImplementedError" in ast.unparse(node):
            pytest.fail(f"{fn.__name__}: not implemented yet")
        if isinstance(node, _LOOPS):
            pytest.fail(f"{fn.__name__}: Python loop or comprehension on line {node.lineno} of the function")
        name = node.id if isinstance(node, ast.Name) else node.attr if isinstance(node, ast.Attribute) else None
        if name in banned:
            pytest.fail(f"{fn.__name__}: `{name}` is off limits in this drill")


@pytest.fixture
def no_loops():
    """no_loops(f, g, ..., ban=("flip",)) reads each function's source and fails if it
    is still the stub, or contains a Python loop, a comprehension, np.vectorize-style
    disguised loops, or any name listed in `ban`."""

    def check(*fns, ban=()):
        for fn in fns:
            _check(fn, _DISGUISED_LOOPS | set(ban))

    return check
