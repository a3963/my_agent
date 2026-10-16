from FlagEmbedding import FlagReranker


def get_reranker(
        model_name="BAAI/bge-reranker-v2-m3"
):
    reranker = FlagReranker(
        model_name,
        use_fp16=False
    )

    return reranker


def rerank_documents(
        reranker,
        chinese_query,
        english_query,
        documents,
        k=5
):

    chinese_pairs = []
    english_pairs = []

    for doc in documents:

        chinese_pairs.append(
            [
                chinese_query,
                doc.page_content
            ]
        )

        english_pairs.append(
            [
                english_query,
                doc.page_content
            ]
        )


    chinese_scores = reranker.compute_score(
        chinese_pairs,
        normalize=True
    )

    english_scores = reranker.compute_score(
        english_pairs,
        normalize=True
    )


    scored_documents = []

    for doc,chinese_score,english_score in zip(
        documents,
        chinese_scores,
        english_scores
    ):

        score = max(
            float(chinese_score),
            float(english_score)
        )

        scored_documents.append(
            (
                score,
                doc
            )
        )


    scored_documents.sort(
        key=lambda x:x[0],
        reverse=True
    )


    result = []

    for score,doc in scored_documents[:k]:

        result.append(doc)

    return result