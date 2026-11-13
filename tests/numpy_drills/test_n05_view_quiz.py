import numpy as np
import pytest

from numpy_drills.n05_view_quiz import MODIFIES_X, SHARES_MEMORY


def fresh():
    return np.arange(12.0).reshape(3, 4)


def test_every_prediction_is_filled_in():
    blanks = [k for d in (SHARES_MEMORY, MODIFIES_X) for k, v in d.items() if not isinstance(v, bool)]
    assert not blanks, f"replace None with True/False for: {blanks}"


@pytest.mark.parametrize("expr", list(SHARES_MEMORY))
def test_shares_memory(expr):
    guess = SHARES_MEMORY[expr]
    assert isinstance(guess, bool), "fill in True or False first"
    x = fresh()
    actual = bool(np.shares_memory(eval(expr, {"np": np, "x": x}), x))
    assert guess == actual, f"`{expr}` {'IS' if actual else 'is NOT'} a view of x"


@pytest.mark.parametrize("stmt", list(MODIFIES_X))
def test_modifies_x(stmt):
    guess = MODIFIES_X[stmt]
    assert isinstance(guess, bool), "fill in True or False first"
    x = fresh()
    exec(stmt, {"np": np, "x": x})
    actual = not np.array_equal(x, fresh())
    assert guess == actual, f"`{stmt}` {'DOES' if actual else 'does NOT'} change x"
