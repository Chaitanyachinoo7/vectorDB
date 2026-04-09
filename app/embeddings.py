from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer


def embed_texts(texts: Iterable[str], *, dim: int) -> list[list[float]]:
    vectorizer = HashingVectorizer(
        n_features=dim,
        alternate_sign=False,
        norm=None,
        ngram_range=(1, 2),
        lowercase=True,
    )
    matrix = vectorizer.transform(list(texts)).astype(np.float32)
    dense = matrix.toarray()

    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    dense = dense / norms

    return dense.tolist()
