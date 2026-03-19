import numpy as np
import torch
import nltk
import faiss
import fitz  # pymupdf
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading nltk punkt...")
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading nltk punkt_tab...")
    nltk.download('punkt_tab')


def load_pdf_chunks(pdf_paths: List[str], chunk_size: int = 500) -> List[Dict]:
    """Extract text from PDFs and split into chunks formatted as collection docs."""
    collection = []
    doc_index = 0
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Could not open {pdf_path}: {e}")
            continue
        for page_num, page in enumerate(doc):
            text = page.get_text()
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size].strip()
                if chunk:
                    collection.append({
                        "id": f"pdf_{doc_index}",
                        "text": chunk,
                        "source": pdf_path,
                        "page": page_num + 1
                    })
                    doc_index += 1
    return collection


class FAISSRetriever:
    """Dense retriever backed by a FAISS index using sentence-transformers."""
    def __init__(self, collection: List[Dict], model_name: str = "all-MiniLM-L6-v2"):
        self.collection = collection
        self.doc_ids = [doc["id"] for doc in collection]
        self.doc_texts = [doc["text"] for doc in collection]

        print(f"Initializing FAISS retriever with {model_name}...")
        self.encoder = SentenceTransformer(model_name)

        print("Encoding documents...")
        embeddings = self.encoder.encode(self.doc_texts, show_progress_bar=True, convert_to_numpy=True)
        embeddings = embeddings.astype("float32")
        faiss.normalize_L2(embeddings)

        self.index = faiss.IndexFlatIP(embeddings.shape[1])  # inner product = cosine after L2 norm
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} vectors.")

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        query_vec = self.encoder.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(query_vec)
        scores, indices = self.index.search(query_vec, k)
        return [(self.doc_ids[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i != -1]


class BaseRetriever:
    """Base class for all retrievers."""
    def __init__(self, collection):
        self.collection = collection
        self.doc_ids = [doc["id"] for doc in collection]
        self.doc_texts = [doc["text"] for doc in collection]

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        raise NotImplementedError

class BM25Retriever(BaseRetriever):
    """Sparse BM25 retriever."""
    def __init__(self, collection):
        super().__init__(collection)
        print("Initializing BM25 retriever...")
        self.tokenized_docs = self._tokenize_docs(self.doc_texts)
        self.bm25 = BM25Okapi(self.tokenized_docs)
        print("BM25 retriever initialized!")

    def _tokenize_docs(self, docs):
        """Lowercase, tokenize and filter non-alphanumeric tokens."""
        tokenized_docs = []
        for doc in tqdm(docs, desc="Tokenizing documents"):
            tokens = word_tokenize(doc.lower())
            tokens = [token for token in tokens if token.isalnum()]
            tokenized_docs.append(tokens)
        return tokenized_docs

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        tokenized_query = word_tokenize(query.lower())
        tokenized_query = [token for token in tokenized_query if token.isalnum()]

        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((self.doc_ids[idx], float(scores[idx])))

        return results

class DenseRetriever(BaseRetriever):
    """Dense retriever using a HuggingFace encoder model with numpy dot-product search."""
    def __init__(self, collection, model_name="BAAI/bge-small-en-v1.5"):
        super().__init__(collection)
        print(f"Initializing Dense retriever with {model_name}...")

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        print("Precomputing document embeddings...")
        self.doc_embeddings = self._encode_docs(self.doc_texts)
        print("Dense retriever initialized!")

    def _encode_docs(self, docs, batch_size=32):
        """Encode documents in batches and return stacked embeddings."""
        embeddings = []
        for i in tqdm(range(0, len(docs), batch_size), desc="Encoding documents"):
            batch_docs = docs[i:i+batch_size]
            inputs = self.tokenizer(
                batch_docs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(batch_embeddings.cpu().numpy())

        return np.vstack(embeddings)

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pool token embeddings weighted by the attention mask."""
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        if "bge" in self.model_name.lower():
            query = f"Represent this sentence for searching relevant passages: {query}"

        inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            query_embedding = self._mean_pooling(outputs, inputs['attention_mask'])
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
            query_embedding = query_embedding.cpu().numpy()

        similarities = np.dot(self.doc_embeddings, query_embedding.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:k]

        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(similarities[idx])))

        return results

class HybridRetriever:
    """Combines multiple retrievers with weighted score fusion."""
    def __init__(self, retrievers: List[BaseRetriever], weights: List[float] = None):
        self.retrievers = retrievers
        self.weights = weights if weights else [1.0] * len(retrievers)
        print(f"Initialized Hybrid retriever with {len(retrievers)} methods")

    def retrieve(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        all_results = defaultdict(float)

        for retriever, weight in zip(self.retrievers, self.weights):
            results = retriever.retrieve(query, k * 2)  # fetch extra candidates before merging

            for doc_id, score in results:
                all_results[doc_id] += score * weight

        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)[:k]
        return sorted_results