import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QwenRAGGenerator:
    """RAG generator backed by a Qwen instruction-tuned model."""
    def __init__(self, model_name="Qwen/Qwen3-0.6B"):
        print(f"Initializing Qwen generator with {model_name}")

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.system_message = "You are a helpful assistant that answers questions based on the provided context."

    def format_context(self, retrieved_docs, collection):
        """Format retrieved docs into a context string for the prompt."""
        doc_map = {doc["id"]: doc["text"] for doc in collection}
        context_parts = []
        for doc_id, score in retrieved_docs:
            doc_text = doc_map.get(doc_id, "")
            if len(doc_text) > 800:
                doc_text = doc_text[:800] + "..."
            context_parts.append(f"Document {doc_id}: {doc_text}")
        return "\n\n".join(context_parts)

    def generate_answer(self, question, context, max_new_tokens=512):
        """Generate an answer using the Qwen model."""
        user_content = f"""Context:
{context}

Question: {question}

Please answer based on the context. If the answer is not in the context, say so."""

        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": user_content}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7
        )

        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Qwen3 wraps internal reasoning in <think>...</think> — capture and strip it
        self.last_thinking = ""
        if "<think>" in response and "</think>" in response:
            start = response.find("<think>") + len("<think>")
            end = response.rfind("</think>")
            self.last_thinking = response[start:end].strip()
            response = response[end + len("</think>"):].strip()

        return response.strip()

class AgenticRAGSystem:
    """Agentic RAG system with planning, decomposition, retrieval and self-check."""

    def __init__(self, collection, retriever, generator):
        self.collection = collection
        self.retriever = retriever
        self.generator = generator
        self.generator.last_thinking = ""
        print("Agentic RAG system initialized")

    def _decompose_complex_query(self, question):
        """Break a complex question into simpler sub-queries."""
        decomposition_prompt = f"""Decompose this complex question into simpler sub-questions:

Original question: {question}

Provide 2-3 simpler questions that together would help answer the original question.
Format each sub-question on a new line starting with "-"."""

        decomposition = self.generator.generate_answer(decomposition_prompt, "", max_new_tokens=400)

        sub_questions = []
        for line in decomposition.split('\n'):
            if line.strip().startswith('-'):
                sub_q = line.strip()[1:].strip()
                if sub_q and len(sub_q) > 10:
                    sub_questions.append(sub_q)

        return sub_questions if sub_questions else [question]

    def _parallel_retrieval(self, queries, k=3):
        """Retrieve documents for each sub-query."""
        all_results = {}
        for query in queries:
            results = self.retriever.retrieve(query, k)
            all_results[query] = results
        return all_results

    def _generate_with_reasoning(self, question, context, sub_queries=None, retrieval_results=None):
        """Generate an answer using a chain-of-thought reasoning prompt."""
        reasoning_prompt = f"""Answer the question using a step-by-step reasoning process.

Question: {question}

Available information:
{context}

Reasoning steps:
1. First, let me identify what information is needed to answer this question.
2. Next, I'll examine the provided context for relevant evidence.
3. Then, I'll synthesize the information to form a complete answer.
4. Finally, I'll verify that the answer is supported by the evidence.

Answer:"""

        reasoning_answer = self.generator.generate_answer(reasoning_prompt, "", max_new_tokens=600)
        return reasoning_answer

    def _self_check_answer(self, question, answer, retrieved_docs, collection):
        """Verify the answer is grounded in the retrieved evidence."""
        doc_map = {doc["id"]: doc["text"] for doc in collection}
        evidence_context = ""
        for doc_id in retrieved_docs[:3]:
            evidence_context += f"Document {doc_id}: {doc_map.get(doc_id, '')}\n\n"

        verification_prompt = f"""Verify if the following answer is supported by the evidence:

Question: {question}
Proposed Answer: {answer}

Evidence:
{evidence_context}

Please check:
1. Is the answer directly supported by the evidence?
2. Are there any unsupported claims in the answer?
3. If parts are unsupported, what should be corrected?

Verification:"""

        verification = self.generator.generate_answer(verification_prompt, "", max_new_tokens=400)

        if "unsupported" in verification.lower() or "not supported" in verification.lower():
            return False, verification
        else:
            return True, verification

    def _reflect_and_refine(self, question, initial_answer, verification_feedback, context):
        """Refine the answer based on self-check feedback."""
        reflection_prompt = f"""Based on the verification feedback, refine the answer:

Original Question: {question}
Initial Answer: {initial_answer}
Verification Feedback: {verification_feedback}
Context: {context}

Please provide an improved answer that addresses the verification concerns:"""

        refined_answer = self.generator.generate_answer(reflection_prompt, "", max_new_tokens=500)
        return refined_answer

    def query(self, question, k=5, enable_planning=True, enable_self_check=True):
        """Run the full agentic query workflow."""

        if enable_planning:
            sub_queries = self._decompose_complex_query(question)
            print(f"Sub-queries: {sub_queries}")

            if len(sub_queries) > 1:
                print("Performing parallel retrieval...")
                all_retrieval_results = self._parallel_retrieval(sub_queries, k=2)

                best_scores = {}
                for query_results in all_retrieval_results.values():
                    for doc_id, score in query_results:
                        if doc_id not in best_scores or score > best_scores[doc_id]:
                            best_scores[doc_id] = score

                retrieved_docs = sorted(best_scores.items(), key=lambda x: x[1], reverse=True)[:k]
                print(f"Merged retrieval results: {len(retrieved_docs)} documents")
            else:
                retrieved_docs = self.retriever.retrieve(question, k)
        else:
            retrieved_docs = self.retriever.retrieve(question, k)
            sub_queries = [question]

        context = self.generator.format_context(retrieved_docs, self.collection)

        print("Generating answer with reasoning...")
        initial_answer = self._generate_with_reasoning(
            question, context, sub_queries, retrieved_docs
        )

        final_answer = initial_answer
        verification_passed = True
        verification_feedback = ""

        if enable_self_check:
            print("Performing self-check...")
            verification_passed, verification_feedback = self._self_check_answer(
                question, initial_answer, [doc_id for doc_id, score in retrieved_docs], self.collection
            )

            print(f"Verification: {'PASS' if verification_passed else 'FAIL'}")

            if not verification_passed:
                print("Refining answer based on verification...")
                final_answer = self._reflect_and_refine(
                    question, initial_answer, verification_feedback, context
                )

        return {
            "answer": final_answer,
            "initial_answer": initial_answer,
            "supporting_docs": [doc_id for doc_id, score in retrieved_docs],
            "retrieval_scores": {doc_id: score for doc_id, score in retrieved_docs},
            "verification_passed": verification_passed,
            "verification_feedback": verification_feedback,
            "sub_queries": sub_queries if enable_planning else [],
        }

