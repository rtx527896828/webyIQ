"""Cosine / max-diff between two logits dumps ([n, vocab] f32), last position and worst position."""

import sys

import numpy as np

V = 130560


def load(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, V)


def cos(x, y):
    return float((x * y).sum() / np.linalg.norm(x) / np.linalg.norm(y))


a, b = load(sys.argv[1]), load(sys.argv[2])
n = min(len(a), len(b))
a, b = a[-n:], b[-n:]
per = [cos(a[i], b[i]) for i in range(n)]
print(f"last: cos {per[-1]:.6f} maxdiff {abs(a[-1] - b[-1]).max():.3f} argmax {a[-1].argmax()} vs {b[-1].argmax()} | "
      f"worst pos {int(np.argmin(per))}: cos {min(per):.6f}")
