import numpy as np
import torch
import nltk
import faiss
import fitz  # pymupdf
from typing import List, Tuple, Dict, Any
from collections import defaultdict
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize, sent_tokenize
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


def load_pdf_chunks(
    pdf_paths: List[str],
    chunk_size_tokens: int = 256,
    overlap_tokens: int = 50,
    tokenizer: Any = None,
    tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> List[Dict]:
    """Extract text from PDFs and split into token-aware chunks.

    Args:
        pdf_paths: list of PDF file paths.
        chunk_size_tokens: target chunk size in tokens.
        overlap_tokens: number of tokens to overlap between adjacent chunks.
        tokenizer: optional HuggingFace tokenizer instance. If None, one is
            created from `tokenizer_model`.
        tokenizer_model: model id used to create a tokenizer when `tokenizer`
            is not provided.

    The function tokenizes sentences and builds chunks by concatenating
    sentences until the token count reaches `chunk_size_tokens`. The window
    then slides back by approximately `overlap_tokens` (measured in tokens)
    to create overlap. This preserves sentence boundaries while remaining
    token-budget aware for embedding models and LLMs.
    """
    # lazily import tokenizer if not passed
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model, use_fast=True)
        except Exception:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)

    collection = []
    doc_index = 0
    for pdf_path in pdf_paths:
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            print(f"Could not open {pdf_path}: {e}")
            continue

        for page_num, page in enumerate(doc):
            text = page.get_text().strip()
            if not text:
                continue

            sentences = sent_tokenize(text)
            # precompute token lengths of sentences for efficiency
            sent_toks = [len(tokenizer.encode(s, add_special_tokens=False)) for s in sentences]

            i = 0
            while i < len(sentences):
                toks = 0
                j = i
                # add sentences until we reach chunk token budget
                while j < len(sentences) and (toks + sent_toks[j] <= chunk_size_tokens or j == i):
                    toks += sent_toks[j]
                    j += 1

                chunk = " ".join(sentences[i:j]).strip()
                if chunk:
                    collection.append({
                        "id": f"pdf_{doc_index}",
                        "text": chunk,
                        "source": pdf_path,
                        "page": page_num + 1,
                    })
                    doc_index += 1

                # slide window back by overlap_tokens measured in sentence steps
                if overlap_tokens <= 0:
                    i = j
                    continue

                # find new start index such that tokens in sentences[new_i:j] <= overlap_tokens
                new_i = j
                overlap_count = 0
                while new_i > i and overlap_count < overlap_tokens:
                    new_i -= 1
                    overlap_count += sent_toks[new_i]

                # ensure forward progress
                if new_i <= i:
                    i = i + 1
                else:
                    i = new_i

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