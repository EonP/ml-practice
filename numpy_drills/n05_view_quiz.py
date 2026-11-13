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
    "x[1]": None,
    "x[:, 1]": None,
    "x[1:3, ::2]": None,
    "x[::-1]": None,
    "x[..., None]": None,
    "x[1:2]": None,
    "x[[1]]": None,
    "x[[0, 2]]": None,
    "x[:, [0]]": None,
    "x[x > 5]": None,
    "x.T": None,
    "x.reshape(4, 3)": None,
    "x[:2].reshape(8)": None,
    "x[:, :2].reshape(6)": None,
    "x.ravel()": None,
    "x.T.ravel()": None,
    "x.T.reshape(12)": None,
    "x.flatten()": None,
    "np.asarray(x)": None,
    "np.array(x)": None,
    "x.astype(np.float64)": None,
    "x.astype(np.float64, copy=False)": None,
    "x + 0": None,
    "np.broadcast_to(x[0], (5, 4))": None,
}

# Each statement runs on a fresh x = np.arange(12.0).reshape(3, 4).
# True if the statement changes x.
MODIFIES_X = {
    "x[0][1] = -1": None,
    "x[:, 0][1] = -1": None,
    "x.T[0] = -1": None,
    "x[[0, 1]] = -1": None,
    "x[[0, 1]][0] = -1": None,
    "x[x > 5] = -1": None,
    "x[x > 5][0] = -1": None,
    "x[1:][[0]] = -1": None,
    "y = x[:, [0, 1]]; y[:] = -1": None,
    "y = x[1:]; y += 1": None,
    "y = x[1:]; y = y + 1": None,
    "y = x.ravel(); y[0] = -1": None,
    "y = x.T.ravel(); y[0] = -1": None,
    "y = x.flatten(); y[0] = -1": None,
    "x.reshape(-1)[0] = -1": None,
    "x.T.reshape(-1)[0] = -1": None,
}
