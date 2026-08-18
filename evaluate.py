# this is the fourth piece of our pipeline:
# a simple evaluator that checks if the model's answers are correct.


import json

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from rag import answer_question

load_dotenv()

# Written after each run so the score can be checked (or shown in the
# README) without re-running the eval — mirrors best_chunk_params.json.
RESULTS_FILE = "evaluate_results.json"

# Questions and real expected answers grounded in the NF-AI documents,
# matching the worked-example and document-owner questions used in
# chunk_experiment.py's TEST_QUERIES.
TESTS = [
    {
        "question": "In the procurement assistant worked example from the Adaptive Agent Orchestration Methodology, what problem did the verifier detect with one supplier's quote?",
        "expected": "The verifier detected that one supplier's quote was expired, so the system requested updated evidence instead of fabricating a comparison.",
    },
    {
        "question": "Who is the primary owner of the Adaptive Agent Orchestration Methodology document?",
        "expected": "Agent Systems Engineering",
    },
    {
        "question": "In the Semantic Context Compression Pipeline worked example, how many messages did the customer-support case span?",
        "expected": "120 messages",
    },
    {
        "question": "Who is the primary owner of the Semantic Context Compression Pipeline document?",
        "expected": "Context Systems Team",
    },
    {
        "question": "In the Grounded RAG Quality Engineering Playbook worked example, what did the employee ask about a vendor storing customer data?",
        "expected": "Whether the vendor can store customer data outside the EU.",
    },
    {
        "question": "Who is the primary owner of the Grounded RAG Quality Engineering Playbook document?",
        "expected": "Knowledge Systems Engineering",
    },
    {
        "question": "In the Prompt Contract Engineering Standard worked example, how many queues does the support triage prompt select from?",
        "expected": "Five queues",
    },
    {
        "question": "Who is the primary owner of the Prompt Contract Engineering Standard document?",
        "expected": "LLM Application Engineering",
    },
    {
        "question": "In the Responsible AI Governance Lifecycle worked example, what does the hiring support tool do?",
        "expected": "It ranks applicants for recruiter review.",
    },
    {
        "question": "Who is the primary owner of the Responsible AI Governance Lifecycle document?",
        "expected": "AI Risk and Governance Office",
    },
    {
        "question": "In the Layered Agent Memory Architecture worked example, what Python version does the project assistant initially learn the team uses?",
        "expected": "Python 3.12",
    },
    {
        "question": "Who is the primary owner of the Layered Agent Memory Architecture document?",
        "expected": "Agent Platform Architecture",
    },
    {
        "question": "In the Compound AI Evaluation Methodology worked example, what kind of unsupported claims did the adversarial cases produce?",
        "expected": "Unsupported medical claims.",
    },
    {
        "question": "Who is the primary owner of the Compound AI Evaluation Methodology document?",
        "expected": "AI Evaluation and Reliability",
    },
    {
        "question": "In the Production LLM Deployment Standard worked example, what type of service uses a primary premium model and a smaller fallback model?",
        "expected": "A summarization service.",
    },
    {
        "question": "Who is the primary owner of the Production LLM Deployment Standard document?",
        "expected": "AI Platform and Site Reliability",
    },
    {
        "question": "In the Hybrid Vector Search Optimization Guide worked example, what is the example search query used to illustrate fusion and reranking?",
        "expected": "ACME-417 retention exception",
    },
    {
        "question": "Who is the primary owner of the Hybrid Vector Search Optimization Guide document?",
        "expected": "Search and Retrieval Engineering",
    },
    {
        "question": "In the Model Context Protocol Integration Blueprint worked example, what two MCP servers does the IDE assistant connect to?",
        "expected": "A source-control MCP server and an internal documentation server.",
    },
    {
        "question": "Who is the primary owner of the Model Context Protocol Integration Blueprint document?",
        "expected": "AI Integration Architecture",
    },
]

JUDGE_PROMPT = """You are grading an answer.
Question: {question}
Reference answer: {expected}
Model answer: {actual}
Does the model answer match the reference in meaning?
Reply with only one word: PASS or FAIL."""


def main():
    judge = init_chat_model("openai:gpt-4o-mini", temperature=0)
    passed = 0
    details = []
    for t in TESTS:
        actual, _ = answer_question(t["question"])
        # the .content.strip().upper() is used to remove any whitespace and
        # convert the verdict to uppercase to match the expected format.
        verdict = judge.invoke(JUDGE_PROMPT.format(
            question=t["question"], expected=t["expected"], actual=actual
        )).content.strip().upper()
        ok = verdict.startswith("PASS") # the ok variable is used to check if the verdict starts with a PASS.
        passed += 1 if ok else 0
        print(f"[{'PASS' if ok else 'FAIL'}] {t['question']}")
        details.append({"question": t["question"], "passed": ok, "actual": actual})

    score = round(passed / len(TESTS), 2)
    print(f"\nScore: {passed}/{len(TESTS)}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(
            {"passed": passed, "total": len(TESTS), "score": score},
            f,
            indent=2,
        )
    print(f"Saved results to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
