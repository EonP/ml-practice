import ast
import inspect
import sys
import textwrap

import pytest

_LOOPS = (ast.For, ast.AsyncFor, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


def _check(obj, banned, loops):
    module = sys.modules[obj.__module__]
    own = {
        name
        for name, value in vars(module).items()
        if (inspect.isfunction(value) or inspect.isclass(value)) and getattr(value, "__module__", None) == module.__name__
    }
    tree = ast.parse(textwrap.dedent(inspect.getsource(obj)))
    for node in ast.walk(tree):
        if isinstance(node, ast.Raise) and "NotImplementedError" in ast.unparse(node):
            pytest.fail(f"{obj.__name__}: not implemented yet")
        if not loops and isinstance(node, _LOOPS):
            pytest.fail(f"{obj.__name__}: Python loop or comprehension on line {node.lineno}; vectorise this one")
        if isinstance(node, ast.Attribute) and node.attr in banned:
            pytest.fail(f"{obj.__name__}: `{node.attr}` is off limits in this drill")
        if isinstance(node, ast.Name) and node.id in banned and node.id not in own:
            pytest.fail(f"{obj.__name__}: `{node.id}` is off limits in this drill")


@pytest.fixture
def no_shortcuts():
    """no_shortcuts(f, SomeClass, ..., ban=("one_hot",), loops=False) reads the source of each
    function or class and fails if it is still the stub, uses a banned name (as `x.name`, or as a
    bare `name` that isn't defined in the drill file itself), or, with loops=False, contains a
    Python loop or comprehension."""

    def check(*objs, ban=(), loops=True):
        for obj in objs:
            _check(obj, set(ban), loops)

    return check
