import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall
)
from ingestion.embedder import embed_query
from retrieval.vector_store import search_vectors
from retrieval.bm25_store import search_bm25
from retrieval.reranker import rerank_chunks
from generation.answer import generate_answer
from langchain_core.documents import Document
from test_dataset import test_data
from ragas.llms import LangchainLLMWrapper
from langchain_groq import ChatGroq
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_huggingface import HuggingFaceEmbeddings


def run_pipeline(question: str):
    """
    Runs the full RAG pipeline for a question.
    Returns answer text and list of retrieved chunk texts.
    """
    # Step 1: Embed query
    query_embedding = embed_query(question)

    # Step 2: Hybrid search
    vector_results = search_vectors(query_embedding, top_k=10)
    bm25_results = search_bm25(question, top_k=10)

    # Step 3: Merge and deduplicate
    merged = []
    for match in vector_results:
        merged.append(Document(
            page_content=match.metadata["text"],
            metadata={
                "filename": match.metadata.get("filename", "unknown"),
                "page": match.metadata.get("page", 0)
            }
        ))
    merged.extend(bm25_results)

    seen = set()
    unique = []
    for doc in merged:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)

    # Step 4: Rerank
    reranked = rerank_chunks(question, unique, top_n=5)

    # Step 5: Generate answer
    result = generate_answer(question, reranked)

    # Extract just the chunk texts for RAGAS
    contexts = [chunk["text"] for chunk in reranked]

    return result["answer"], contexts


def build_ragas_dataset():
    """
    Runs pipeline on all test questions and builds
    the dataset structure RAGAS expects.
    """
    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, item in enumerate(test_data):
        print(f"Running question {i+1}/{len(test_data)}: {item['question'][:60]}...")

        try:
            answer, retrieved_contexts = run_pipeline(item["question"])

            questions.append(item["question"])
            answers.append(answer)
            contexts.append(retrieved_contexts)  # list of strings
            ground_truths.append(item["ground_truth"])

        except Exception as e:
            print(f"Failed on question {i+1}: {e}")
            # Add empty results so dataset stays aligned
            questions.append(item["question"])
            answers.append("")
            contexts.append([""])
            ground_truths.append(item["ground_truth"])

    return Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })


def main():
    print("Building evaluation dataset...")
    dataset = build_ragas_dataset()
    groq_llm = LangchainLLMWrapper(ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    n=1
    ))

    hf_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings( 
    model_name="BAAI/bge-base-en"
    ))

    results = evaluate(
    dataset,
    metrics=[
    Faithfulness(),
    AnswerRelevancy(),
    ContextPrecision(),
    ContextRecall()
],
    llm=groq_llm,
    embeddings=hf_embeddings
) 

    
        

        # Replace your print block with this
    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)

    df = results.to_pandas()
    print(df[['faithfulness', 'answer_relevancy', 
            'context_precision', 'context_recall']].mean())

    faith = df['faithfulness'].mean()
    relevancy = df['answer_relevancy'].mean()
    precision = df['context_precision'].mean()
    recall = df['context_recall'].mean()

    print("="*50)
    print(f"Faithfulness:      {faith:.4f}")
    print(f"Answer Relevancy:  {relevancy:.4f}")
    print(f"Context Precision: {precision:.4f}")
    print(f"Context Recall:    {recall:.4f}")
    print("="*50)

    # Save to file
    with open("evaluation_results.txt", "w") as f:
        f.write("RAGAS Evaluation Results\n")
        f.write("="*50 + "\n")
        f.write(f"Faithfulness:      {faith:.4f}\n")
        f.write(f"Answer Relevancy:  {relevancy:.4f}\n")
        f.write(f"Context Precision: {precision:.4f}\n")
        f.write(f"Context Recall:    {recall:.4f}\n")
        f.write(f"Total questions:   {len(test_data)}\n")

        print("\nResults saved to evaluation_results.txt")
        return results


if __name__ == "__main__":
    main()