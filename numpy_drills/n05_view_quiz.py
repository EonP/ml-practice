"""NumPy drill 05: view or copy? (target: 8 min). No code: predictions only.

Basic indexing (integers, slices, ..., None) returns a VIEW that shares memory
with the original. Advanced indexing (integer arrays, boolean masks) returns a
COPY. Some functions return a view when they can and a copy when they must.

Replace every None with True or False BEFORE running anything. Then run the
tests: each failure names an expression you got wrong. Work out why before
moving on.
"""

# x = np.arange(12.0).reshape(3, 4)
# True if the result shares memory with x, i.e. np.shares_memory(result, x).
SHARES_MEMORY = {
    "x[1]": True,
    "x[:, 1]": True,
    "x[1:3, ::2]": True,
    "x[::-1]": True,
    "x[..., None]": True,
    "x[1:2]": True,
    "x[[1]]": False,
    "x[[0, 2]]": False,
    "x[:, [0]]": False,
    "x[x > 5]": False,
    "x.T": True,
    "x.reshape(4, 3)": True,
    "x[:2].reshape(8)": True,
    "x[:, :2].reshape(6)": False,
    "x.ravel()": True,
    "x.T.ravel()": False,
    "x.T.reshape(12)": False,
    "x.flatten()": False,
    "np.asarray(x)": True,
    "np.array(x)": False,
    "x.astype(np.float64)": False,
    "x.astype(np.float64, copy=False)": True,
    "x + 0": False,
    "np.broadcast_to(x[0], (5, 4))": True,
}

# Each statement runs on a fresh x = np.arange(12.0).reshape(3, 4).
# True if the statement changes x.
MODIFIES_X = {
    "x[0][1] = -1": True,
    "x[:, 0][1] = -1": True,
    "x.T[0] = -1": True,
    "x[[0, 1]] = -1": True,
    "x[[0, 1]][0] = -1": False,
    "x[x > 5] = -1": True,
    "x[x > 5][0] = -1": False,
    "x[1:][[0]] = -1": True,
    "y = x[:, [0, 1]]; y[:] = -1": False,
    "y = x[1:]; y += 1": True,
    "y = x[1:]; y = y + 1": False,
    "y = x.ravel(); y[0] = -1": True,
    "y = x.T.ravel(); y[0] = -1": False,
    "y = x.flatten(); y[0] = -1": False,
    "x.reshape(-1)[0] = -1": True,
    "x.T.reshape(-1)[0] = -1": False,
}
