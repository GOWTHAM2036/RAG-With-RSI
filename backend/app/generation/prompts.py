"""
Prompt templates for RAG generation and agent self-reflection.
"""

CONCISE_PROMPT = """You are EvoRAG, an intelligent AI research assistant.
Answer the question concisely based STRICTLY on the provided context. If the context does not contain enough information, state that clearly.

Context:
{context}

Question: {query}
Answer:"""

COT_PROMPT = """You are EvoRAG, an intelligent AI research assistant.
Analyze the provided context carefully and reason step-by-step before delivering the final answer.

Context:
{context}

Question: {query}

Thinking Process:
1. Identify key entities and facts in the context related to the query.
2. Verify relationships between facts.
3. Formulate the final grounded answer.

Answer:"""

CITATION_STRICT_PROMPT = """You are EvoRAG, a precision knowledge synthesis engine.
Synthesize a comprehensive answer to the user query using the numbered context chunks below.
Every factual assertion MUST be cited with the corresponding bracketed number, e.g. [1] or [2].
Do not invent information outside the provided sources.

Context Chunks:
{context}

User Query: {query}

Synthesized Answer with Citations:"""

QUERY_REWRITE_PROMPT = """Rewrite the user's input into a standalone, search-optimized search query.
Resolve ambiguous pronouns and extract core concepts.

Original Query: {query}
Optimized Search Query:"""

FAITHFULNESS_CHECK_PROMPT = """Evaluate if the generated answer is strictly grounded in the context without hallucinations.
Context:
{context}

Answer:
{answer}

Output JSON with keys:
- grounded: true/false
- hallucination_score: float between 0.0 and 1.0 (0.0 = completely faithful, 1.0 = completely fabricated)
- explanation: short reasoning"""

def format_context_chunks(chunks) -> str:
    formatted = []
    for idx, r_chunk in enumerate(chunks):
        title = r_chunk.chunk.metadata.get("doc_title", "Document")
        formatted.append(f"[{idx + 1}] Source: {title}\n{r_chunk.chunk.content.strip()}")
    return "\n\n".join(formatted)

def get_prompt_by_template_name(template_name: str) -> str:
    t = template_name.lower().strip()
    if t in ["cot", "chain_of_thought"]:
        return COT_PROMPT
    elif t in ["concise", "brief"]:
        return CONCISE_PROMPT
    else:
        return CITATION_STRICT_PROMPT
