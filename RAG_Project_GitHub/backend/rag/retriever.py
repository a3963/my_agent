def search_document(vectorstore,query,k = 3):
    docs = vectorstore.similarity_search(query,k = k)
    return docs

def search_document_diverse(
        vectorstore,
        query,
        fetch_k=10,
        k=3
):
    documents = vectorstore.similarity_search(
        query,
        k=fetch_k
    )

    result = []
    seen_sources = set()

    for doc in documents:

        source = doc.metadata.get(
            'source',
            ''
        )

        if source in seen_sources:
            continue

        seen_sources.add(source)

        result.append(doc)

        if len(result) >= k:
            break

    return result


def search_document_bilingual(
        vectorstore,
        chinese_query,
        english_query,
        fetch_k=20,
        k=3,
        chinese_weight=1.0,
        english_weight=1.0,
        rrf_k=60
):

    chinese_documents = vectorstore.similarity_search(
        chinese_query,
        k=fetch_k
    )

    english_documents = vectorstore.similarity_search(
        english_query,
        k=fetch_k
    )

    scores = {}
    source_docs = {}


    chinese_seen = set()

    for rank,doc in enumerate(
        chinese_documents,
        start=1
    ):

        source = doc.metadata.get(
            'source',
            ''
        )

        if source in chinese_seen:
            continue

        chinese_seen.add(source)

        if source not in scores:
            scores[source] = 0
            source_docs[source] = doc

        scores[source] += chinese_weight / (
            rrf_k + rank
        )


    english_seen = set()

    for rank,doc in enumerate(
        english_documents,
        start=1
    ):

        source = doc.metadata.get(
            'source',
            ''
        )

        if source in english_seen:
            continue

        english_seen.add(source)

        if source not in scores:
            scores[source] = 0
            source_docs[source] = doc

        scores[source] += english_weight / (
            rrf_k + rank
        )


    sorted_sources = sorted(
        scores,
        key=scores.get,
        reverse=True
    )


    result = []

    for source in sorted_sources[:k]:
        result.append(
            source_docs[source]
        )

    return result

def search_chunk_bilingual(
        vectorstore,
        chinese_query,
        english_query,
        fetch_k=20,
        k=5,
        chinese_weight=1.0,
        english_weight=1.0,
        rrf_k=60
):

    chinese_documents = vectorstore.similarity_search(
        chinese_query,
        k=fetch_k
    )

    english_documents = vectorstore.similarity_search(
        english_query,
        k=fetch_k
    )

    scores = {}
    chunk_docs = {}


    for rank,doc in enumerate(
        chinese_documents,
        start=1
    ):

        key = (
            doc.metadata.get('source',''),
            doc.metadata.get('页数',0),
            doc.page_content
        )

        if key not in scores:
            scores[key] = 0
            chunk_docs[key] = doc

        scores[key] += chinese_weight / (
            rrf_k + rank
        )


    for rank,doc in enumerate(
        english_documents,
        start=1
    ):

        key = (
            doc.metadata.get('source',''),
            doc.metadata.get('页数',0),
            doc.page_content
        )

        if key not in scores:
            scores[key] = 0
            chunk_docs[key] = doc

        scores[key] += english_weight / (
            rrf_k + rank
        )


    sorted_chunks = sorted(
        scores,
        key=scores.get,
        reverse=True
    )


    result = []

    for key in sorted_chunks[:k]:
        result.append(
            chunk_docs[key]
        )

    return result