from sentence_transformers import SentenceTransformer
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter

# Initialize the Sentence Transformer model
embedder = SentenceTransformer('all-MiniLM-L6-v2')
EMBED_DIM = 384  # Updated dimension for Sentence Transformers

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    # Generate embeddings locally using Sentence Transformers
    embeddings = embedder.encode(texts)
    return embeddings.tolist()