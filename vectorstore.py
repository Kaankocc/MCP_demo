# app/vectorstore.py
from pinecone import Pinecone
from config import PINECONE_API_KEY
from embedding import CustomEmbeddings

pc = Pinecone(api_key=PINECONE_API_KEY)

# Initialize or get the index
try:
    index = pc.Index("rag-index1")
except Exception:
    # If index doesn't exist, create it
    pc.create_index(
        name="rag-index1",
        dimension=1536,  # This should match your embedding model's dimension
        metric="cosine"
    )
    index = pc.Index("rag-index1")

embedding = CustomEmbeddings()

def query_response(parsed_dict, top_k=4):
    vector = embedding.embed_query(parsed_dict["content_string_query"])

    filters = {}
    if parsed_dict["industry_filter"]:
        filters['Industry Sectors'] = {"$in": parsed_dict["industry_filter"]}
    if parsed_dict["takeaways_filter"]:
        filters['Takeaways'] = {"$in": parsed_dict["takeaways_filter"]}

    return index.query(
        vector=vector,
        top_k=top_k,
        include_values=True,
        include_metadata=True,
        filter=filters or None
    )

