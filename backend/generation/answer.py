from groq import Groq
from config import GROQ_API_KEY

client = Groq(api_key=GROQ_API_KEY)

def generate_answer(query: str, reranked_chunks: list):
    if not reranked_chunks:
        return {
            "answer": "No relevant content found in uploaded documents.",
            "citations": []
        }

    # Build context from top chunks
    context = ""
    citations = []
    for i, chunk in enumerate(reranked_chunks):
        context += f"[Source {i+1}] {chunk['text']}\n\n"
        citations.append({
            "source": chunk["filename"],
            "page": chunk["page"]
        })

    prompt = f"""You are an exam prep assistant for engineering students.
Answer the question using ONLY the provided context.
Always cite which source your answer comes from using [Source N].
If the answer is not in the context, say "I couldn't find this in your uploaded documents."

Context:
{context}

Question: {query}

Answer:"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2  # Low temperature = more factual
    )

    return {
        "answer": response.choices[0].message.content,
        "citations": citations
    }