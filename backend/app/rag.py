import uuid
import time
import requests
from typing import List
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding 
from sentence_transformers import CrossEncoder

from app.config import QDRANT_URL, TEI_URL, COLLECTION_NAME, DENSE_VECTOR_SIZE
from app.prepare_data import markdown_documents

client = QdrantClient(url=QDRANT_URL)
sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1", threads=4)
reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def setup_collection():
    """Initialize Qdrant with hybrid vectors"""
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense-vector": models.VectorParams(
                size=DENSE_VECTOR_SIZE,  
                distance=models.Distance.COSINE
            )
        },
        sparse_vectors_config={
            "sparse-vector": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False)
            )
        }
    )

def get_dense_embedding(text: str) -> List[float]:
    response = requests.post(
        TEI_URL, 
        json={"inputs": [text], "normalize": True},
        timeout=2
    )
    response.raise_for_status() 
    return response.json()[0]

def delete_collection():
    try:
        if client.collection_exists(collection_name=COLLECTION_NAME):
            client.delete_collection(collection_name=COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' deleted successfully.")
        else:
            print(f"Collection '{COLLECTION_NAME}' does not exist. Skipping deletion.")
    except Exception as e:
        print(f"An error occurred while trying to delete collection '{COLLECTION_NAME}': {e}")

def wait_for_tei(timeout=102000):
    print("Waiting for TEI model to load...")
    start = time.time()
    while True:
        try:
            r = requests.post(
                TEI_URL,
                json={"inputs": ["hello"], "normalize": True},
                timeout=2
            ) 
            if r.status_code == 200:
                print("✔ TEI model loaded and ready.")
                return
        except Exception:
            pass
        if time.time() - start > timeout:
            raise TimeoutError("TEI model did not start within the expected time.")
        print("… still loading model, waiting 3 seconds...")
        time.sleep(3)

def ingest_documents(markdown_documents: List[str]):
    """Ingest documents with dual embeddings"""
    points_to_upload = []
        
    for i, doc in enumerate(markdown_documents):
        # Dense embedding via TEI
        dense_vector = get_dense_embedding(doc)
                
        # Sparse embedding via SPLADE
        sparse_vector_output = list(sparse_model.query_embed(doc))[0]
        sparse_vector = models.SparseVector(
            indices=sparse_vector_output.indices.tolist(),
            values=sparse_vector_output.values.tolist()
        )
                
        points_to_upload.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector={
                "dense-vector": dense_vector,
                "sparse-vector": sparse_vector
            },
            payload={"text": doc, "chunk_id": i}
        ))
        
    client.upsert(collection_name=COLLECTION_NAME, points=points_to_upload)

def hybrid_rag_query(query_text: str, top_k: int = 16, rerank_top_k: int = 8):
    """Hybrid retrieval + cross-encoder reranking"""
        
    # Get both dense and sparse query vectors
    query_dense = get_dense_embedding(query_text)
    query_sparse_gen = list(sparse_model.embed(documents=[query_text]))[0]
    
    query_sparse = models.SparseVector(
        indices=query_sparse_gen.indices.tolist(),
        values=query_sparse_gen.values.tolist()
    )
        
    # Hybrid search with RRF fusion
    search_results = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(query=query_dense, using="dense-vector", limit=top_k),
            models.Prefetch(query=query_sparse, using="sparse-vector", limit=top_k),
        ],
        query=models.RrfQuery(rrf=models.Rrf(k=60)),
        limit=top_k
    ).points
        
    # Rerank with cross-encoder
    doc_texts = [hit.payload['text'] for hit in search_results]
    cross_encoder_inputs = [[query_text, doc] for doc in doc_texts]
    scores = reranker.predict(cross_encoder_inputs)
        
    # Sort by cross-encoder scores
    ranked_results = sorted(
        [{"text": hit.payload['text'], "score": scores[i]} for i, hit in enumerate(search_results)],
        key=lambda x: x["score"],
        reverse=True
    )
        
    final_docs = [res["text"] for res in ranked_results[:rerank_top_k]]
    context_str = "\n\n---\n\n".join(final_docs)
        
    return context_str, ranked_results[:rerank_top_k]

def ingestion_pipeline():
    try:
        print("Deleting existing collection...")
        delete_collection()
        print("Setting up new collection...")
        setup_collection()
        print("Waiting for TEI model to be ready...")
        wait_for_tei()
        print(f"Ingesting {len(markdown_documents)} markdown documents...")
        ingest_documents(markdown_documents)
        print("Ingestion pipeline completed successfully!")
    except Exception as e:
        print(f"Ingestion pipeline failed: {e}")
        raise
