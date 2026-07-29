import json
import os
import shutil
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from ingest import load_documents, DOCS_DIR

load_dotenv()

# Grid of parameters to try. Add/remove values to widen or narrow the search.
CHUNK_SIZES = [400, 800, 1200]
CHUNK_OVERLAPS = [0, 100, 200]

# Questions grounded in the worked examples of each NF-AI methodology
# document, paired with keywords/phrases that should appear in a chunk that
# correctly answers each question. This is what "hit_rate" is measured against.
TEST_QUERIES = [
    {
        "question": "In the procurement assistant worked example from the Adaptive Agent Orchestration Methodology, what problem did the verifier detect with one supplier's quote?",
        "expected_keywords": ["expired"],
    },
    {
        "question": "Who is the primary owner of the Adaptive Agent Orchestration Methodology document?",
        "expected_keywords": ["Agent Systems Engineering"],
    },
    {
        "question": "In the Semantic Context Compression Pipeline worked example, how many messages did the customer-support case span?",
        "expected_keywords": ["120 messages", "120"],
    },
    {
        "question": "Who is the primary owner of the Semantic Context Compression Pipeline document?",
        "expected_keywords": ["Context Systems Team"],
    },
    {
        "question": "In the Grounded RAG Quality Engineering Playbook worked example, what did the employee ask about a vendor storing customer data?",
        "expected_keywords": ["outside the EU"],
    },
    {
        "question": "Who is the primary owner of the Grounded RAG Quality Engineering Playbook document?",
        "expected_keywords": ["Knowledge Systems Engineering"],
    },
    {
        "question": "In the Prompt Contract Engineering Standard worked example, how many queues does the support triage prompt select from?",
        "expected_keywords": ["five queues", "five"],
    },
    {
        "question": "Who is the primary owner of the Prompt Contract Engineering Standard document?",
        "expected_keywords": ["LLM Application Engineering"],
    },
    {
        "question": "In the Responsible AI Governance Lifecycle worked example, what does the hiring support tool do?",
        "expected_keywords": ["ranks applicants", "recruiter review"],
    },
    {
        "question": "Who is the primary owner of the Responsible AI Governance Lifecycle document?",
        "expected_keywords": ["AI Risk and Governance Office"],
    },
    {
        "question": "In the Layered Agent Memory Architecture worked example, what Python version does the project assistant initially learn the team uses?",
        "expected_keywords": ["Python 3.12", "3.12"],
    },
    {
        "question": "Who is the primary owner of the Layered Agent Memory Architecture document?",
        "expected_keywords": ["Agent Platform Architecture"],
    },
    {
        "question": "In the Compound AI Evaluation Methodology worked example, what kind of unsupported claims did the adversarial cases produce?",
        "expected_keywords": ["medical claims", "medical"],
    },
    {
        "question": "Who is the primary owner of the Compound AI Evaluation Methodology document?",
        "expected_keywords": ["AI Evaluation and Reliability"],
    },
    {
        "question": "In the Production LLM Deployment Standard worked example, what type of service uses a primary premium model and a smaller fallback model?",
        "expected_keywords": ["summarization"],
    },
    {
        "question": "Who is the primary owner of the Production LLM Deployment Standard document?",
        "expected_keywords": ["AI Platform and Site Reliability"],
    },
    {
        "question": "In the Hybrid Vector Search Optimization Guide worked example, what is the example search query used to illustrate fusion and reranking?",
        "expected_keywords": ["ACME-417"],
    },
    {
        "question": "Who is the primary owner of the Hybrid Vector Search Optimization Guide document?",
        "expected_keywords": ["Search and Retrieval Engineering"],
    },
    {
        "question": "In the Model Context Protocol Integration Blueprint worked example, what two MCP servers does the IDE assistant connect to?",
        "expected_keywords": ["source-control", "documentation server"],
    },
    {
        "question": "Who is the primary owner of the Model Context Protocol Integration Blueprint document?",
        "expected_keywords": ["AI Integration Architecture"],
    },
]

# TOP_K values to test for each chunk_size/overlap combo. Retrieving more
# chunks per query can only help hit_rate (more chances to catch the right
# evidence), but it also means more tokens and more irrelevant text fed to
# the generator. The goal is the smallest k that still gets the evidence in.
TOP_K_VALUES = [2, 4, 6, 8]

TEMP_DB_DIR = "chroma_db_experiment"

# ingest.py reads this file to automatically pick up whatever this script
# finds, instead of someone hand-copying numbers into ingest.py.
BEST_PARAMS_FILE = "best_chunk_params.json"


def evaluate_combo(docs, chunk_size, chunk_overlap, embeddings):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    lengths = [len(c.page_content) for c in chunks]
    stats = {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "num_chunks": len(chunks),
        "avg_len": round(sum(lengths) / len(lengths), 1),
        "min_len": min(lengths),
        "max_len": max(lengths),
    }

    if os.path.exists(TEMP_DB_DIR):
        shutil.rmtree(TEMP_DB_DIR)
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=TEMP_DB_DIR,
    )

    # Retrieve once at the largest k and slice it for every smaller k, instead
    # of re-querying the index per k value.
    max_k = max(TOP_K_VALUES)
    per_query_results = [
        (item, db.similarity_search(item["question"], k=max_k))
        for item in TEST_QUERIES
    ]

    hit_rate_by_k = {}
    for k in TOP_K_VALUES:
        hits = 0
        for item, results in per_query_results:
            combined_text = " ".join(r.page_content.lower() for r in results[:k])
            if any(kw.lower() in combined_text for kw in item["expected_keywords"]):
                hits += 1
        hit_rate_by_k[k] = round(hits / len(TEST_QUERIES), 2)

    shutil.rmtree(TEMP_DB_DIR)

    best_hit_rate = max(hit_rate_by_k.values())
    # Smallest k that reaches this combo's best achievable hit_rate: going
    # past this point only adds tokens and noise, not correctness.
    optimal_k = min(k for k, hr in hit_rate_by_k.items() if hr == best_hit_rate)

    stats["hit_rate_by_k"] = hit_rate_by_k
    stats["best_hit_rate"] = best_hit_rate
    stats["optimal_k"] = optimal_k
    return stats


def main():
    if not TEST_QUERIES:
        raise ValueError("Add at least one entry to TEST_QUERIES before running.")

    docs = load_documents(DOCS_DIR)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    results = []
    for chunk_size in CHUNK_SIZES:
        for chunk_overlap in CHUNK_OVERLAPS:
            if chunk_overlap >= chunk_size:
                continue
            print(f"Testing chunk_size={chunk_size}, chunk_overlap={chunk_overlap}...")
            results.append(evaluate_combo(docs, chunk_size, chunk_overlap, embeddings))

    # Best achievable hit_rate first; among ties, prefer the smaller optimal_k
    # (less context sent to the generator), then fewer chunks (cheaper index).
    results.sort(key=lambda r: (-r["best_hit_rate"], r["optimal_k"], r["num_chunks"]))

    k_headers = "".join(f"hit@{k:<6}" for k in TOP_K_VALUES)
    print(f"\n{'chunk_size':<12}{'overlap':<10}{'num_chunks':<12}{'avg_len':<10}{k_headers}{'opt_k':<8}")
    for r in results:
        k_cells = "".join(f"{r['hit_rate_by_k'][k]:<10}" for k in TOP_K_VALUES)
        print(f"{r['chunk_size']:<12}{r['chunk_overlap']:<10}{r['num_chunks']:<12}"
              f"{r['avg_len']:<10}{k_cells}{r['optimal_k']:<8}")

    best = results[0]
    print(f"\nBest combo: chunk_size={best['chunk_size']}, chunk_overlap={best['chunk_overlap']}, "
          f"top_k={best['optimal_k']} (hit_rate={best['best_hit_rate']}, {best['num_chunks']} chunks)")

    with open(BEST_PARAMS_FILE, "w") as f:
        json.dump(
            {
                "chunk_size": best["chunk_size"],
                "chunk_overlap": best["chunk_overlap"],
                "top_k": best["optimal_k"],
                "hit_rate": best["best_hit_rate"],
                "num_chunks": best["num_chunks"],
            },
            f,
            indent=2,
        )
    print(f"Saved winning parameters to {BEST_PARAMS_FILE}")


if __name__ == "__main__":
    main()
