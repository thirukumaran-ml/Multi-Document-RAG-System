import numpy as np


def retrieve_relevant_chunks(
        query,
        model,
        index,
        chunks,
        top_k=3
):
    """
    Retrieve most relevant chunks.
    """

    query_embedding = model.encode([query])

    distances, indices = index.search(
        np.array(query_embedding).astype("float32"),
        top_k
    )

    retrieved_chunks = [
        chunks[i]
        for i in indices[0]
    ]

    return retrieved_chunks