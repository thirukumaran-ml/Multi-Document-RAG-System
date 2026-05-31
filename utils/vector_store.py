import faiss
import numpy as np
import pickle


def create_faiss_index(embeddings):

    embeddings = np.array(
        embeddings
    ).astype(
        "float32"
    )

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(
        embeddings
    )

    return index


def save_faiss_index(
        index,
        path
):

    faiss.write_index(
        index,
        path
    )


def load_faiss_index(
        path
):

    return faiss.read_index(
        path
    )


def save_chunks(
        chunks,
        path
):

    with open(
        path,
        "wb"
    ) as f:

        pickle.dump(
            chunks,
            f
        )


def load_chunks(
        path
):

    with open(
        path,
        "rb"
    ) as f:

        return pickle.load(
            f
        )