"""
ACARA Evaluation Benchmark
===========================
Measures 4 metrics comparing Baseline RAG vs ACARA pipeline:

  1. Answer Accuracy     — GPT-4o-mini judge + optional exact match
  2. Hallucination Rate  — Validator flag rate (Critic/Validator node)
  3. Retrieval Precision — relevant chunks / total chunks retrieved
  4. Latency            — avg response time (baseline vs ACARA)

Test set: pulled from the MedQA / MedQuAD data already ingested into
          the vector store.  Provide a JSONL file or use the built-in
          sample questions when none is supplied.

Usage:
    cd backend
    python eval_benchmark.py                          # uses built-in 500-Q set
    python eval_benchmark.py --dataset path/to/qs.jsonl --n 500
    python eval_benchmark.py --mode exact             # exact-match instead of LLM judge
"""

import os
import sys
import json
import time
import asyncio
import argparse
import logging
from statistics import mean
from datetime import datetime
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams
from typing import Any

from dotenv import load_dotenv

load_dotenv()

# ── stdlib logging ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,           # suppress verbose LangGraph logs
    format="%(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("eval")
logger.setLevel(logging.INFO)

# ─────────────────────────────────────────────────────────────────────────────
# Built-in 500-question test set (imported from eval_testset.py)
# ─────────────────────────────────────────────────────────────────────────────
from eval_testset import EVAL_TEST_SET as SAMPLE_TEST_SET  # noqa: E402



# ─────────────────────────────────────────────────────────────────────────────
# LLM Judge (GPT-4o-mini)
# ─────────────────────────────────────────────────────────────────────────────
from langchain_openai import ChatOpenAI

judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# Initialize DeepEval Metric to mimic RAGAS Answer Correctness
answer_correctness_metric = GEval(
    name="RAGAS Answer Correctness",
    evaluation_steps=[
        "Identify the core facts and concepts present in the expected output.",
        "Check if the actual output contains those core facts.",
        "If the actual output contains the core facts from the expected output, award a score of 1.0.",
        "Do NOT penalize the actual output for having additional facts, being longer, or being more detailed."
    ],
    evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
    strict_mode=False,
    model="gpt-4o-mini"
)

async def llm_judge(question: str, ground_truth: str, candidate_answer: str) -> bool:
    """
    Uses DeepEval GEval metric to score clinical equivalence. Returns True / False.
    """
    test_case = LLMTestCase(
        input=question,
        actual_output=candidate_answer,
        expected_output=ground_truth
    )
    
    try:
        await answer_correctness_metric.a_measure(test_case)
        return answer_correctness_metric.score >= 0.5
    except Exception as e:
        logger.warning(f"DeepEval measure error: {e}")
        return False


def exact_match(ground_truth: str, candidate_answer: str) -> bool:
    """Case-insensitive substring match of ground_truth keywords in answer."""
    keywords = [kw.strip().lower() for kw in ground_truth.split(",")]
    candidate_lower = candidate_answer.lower()
    return all(kw in candidate_lower for kw in keywords)


# ─────────────────────────────────────────────────────────────────────────────
# Baseline RAG (fixed params, no ARC, no Validator)
# ─────────────────────────────────────────────────────────────────────────────
from database import search_vector_store, USE_PINECONE, get_embeddings_model
embeddings_model = get_embeddings_model()

BASELINE_THRESHOLD = 0.5
BASELINE_TOP_K = 5


def baseline_rag(question: str) -> dict[str, Any]:
    """
    Fixed-param RAG — mirrors the ACARA retrieve+generate path but:
      • top_k fixed at 5
      • similarity_threshold fixed at 0.5
      • No ARC feedback loop
      • No Critic/Validator node
    Returns {"answer": str, "chunks_retrieved": int, "relevant_chunks": int}
    """
    query_emb = embeddings_model.embed_query(question) if USE_PINECONE else None
    results = search_vector_store(question, n_results=BASELINE_TOP_K, embedding=query_emb)

    docs = results.get("documents", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    # Filter by threshold
    relevant_docs = []
    for doc, dist in zip(docs, distances):
        if USE_PINECONE:
            score = 1 - dist  # cosine similarity
        else:
            score = 1 / (1 + dist)
        if score >= BASELINE_THRESHOLD:
            relevant_docs.append(doc)

    context = "\n\n---\n\n".join(relevant_docs) if relevant_docs else "\n\n---\n\n".join(docs)

    system = (
        "You are a medical AI assistant. Answer using ONLY the provided context. "
        "If the context does not contain the answer, say 'I don't know'.\n\n"
        f"Context:\n{context}"
    )
    from langchain_core.messages import SystemMessage, HumanMessage
    response = judge_llm.invoke([SystemMessage(content=system), HumanMessage(content=question)])

    return {
        "answer": response.content,
        "chunks_retrieved": len(docs),
        "relevant_chunks": len(relevant_docs),
        "contexts": docs,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ACARA pipeline wrapper (calls agent.process_message)
# ─────────────────────────────────────────────────────────────────────────────
import uuid

# Suppress LangGraph internal logs
logging.getLogger("langgraph").setLevel(logging.ERROR)
logging.getLogger("langchain").setLevel(logging.ERROR)


async def acara_pipeline(question: str) -> dict[str, Any]:
    """
    Full ACARA pipeline: ARC + Validator + CAG.
    Returns {"answer": str, "hallucination_flagged": bool,
             "chunks_retrieved": int, "relevant_chunks": int, "arc_params": dict}
    """
    from arc import arc
    arc.reset()   # fresh ARC state per question to ensure fair comparison

    from agent import process_message
    session_id = f"eval-{uuid.uuid4().hex[:8]}"
    result = await process_message(session_id, question)

    answer = result.get("content", "")
    hallucination_flagged = "⚠️ **Validator Warning**" in answer
    arc_params = result.get("arc_params", {})

    # Retrieve chunk stats using the ARC params that were actually used
    from arc import arc as arc_controller
    top_k = arc_params.get("top_k", 3)
    threshold = arc_params.get("similarity_threshold", 0.45)

    query_emb = embeddings_model.embed_query(question) if USE_PINECONE else None
    results = search_vector_store(question, n_results=top_k, embedding=query_emb)
    docs = results.get("documents", [[]])[0] or []
    distances = results.get("distances", [[]])[0] or []

    relevant_chunks = 0
    for dist in distances:
        score = (1 - dist) if USE_PINECONE else (1 / (1 + dist))
        if score >= threshold:
            relevant_chunks += 1

    return {
        "answer": answer,
        "hallucination_flagged": hallucination_flagged,
        "chunks_retrieved": len(docs),
        "relevant_chunks": relevant_chunks,
        "arc_params": arc_params,
        "contexts": docs,
    }



async def run_eval(test_set: list[dict], mode: str = "llm", output_path: str = None):
    """
    Run full evaluation over `test_set`.
    Measures:
      1. Answer Accuracy     — GPT-4o-mini judge (mode='llm') or exact match (mode='exact')
      2. Hallucination Rate  — Validator flag rate from ACARA pipeline
      3. Retrieval Precision — relevant chunks / total chunks retrieved
      4. Latency            — avg response time (baseline vs ACARA)
    """
    scorer = llm_judge if mode == "llm" else exact_match

    total = len(test_set)
    logger.info(f"Starting evaluation on {total} questions | mode={mode}")
    logger.info("=" * 60)

    b_correct, a_correct = [], []
    b_latency, a_latency = [], []
    b_precision, a_precision = [], []
    a_hallucination = []
    per_question_log = []

    for i, item in enumerate(test_set, 1):
        question     = item["question"]
        ground_truth = item["answer"]

        print(f"  [{i:3d}/{total}] {question[:65]}...", flush=True)

        # ── Baseline ─────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            b = baseline_rag(question)
        except Exception as e:
            logger.warning(f"  Baseline error: {e}")
            b = {"answer": "Error", "chunks_retrieved": 0, "relevant_chunks": 0, "contexts": []}
        b_lat = time.perf_counter() - t0
        b_prec = (b["relevant_chunks"] / b["chunks_retrieved"]) if b["chunks_retrieved"] else 0.0

        if mode == "llm":
            b_ok = await scorer(question, ground_truth, b["answer"])
        else:
            b_ok = scorer(ground_truth, b["answer"])

        b_correct.append(b_ok)
        b_latency.append(b_lat)
        b_precision.append(b_prec)

        # ── ACARA ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        try:
            a = await acara_pipeline(question)
        except Exception as e:
            logger.warning(f"  ACARA error: {e}")
            a = {"answer": "Error", "hallucination_flagged": False,
                 "chunks_retrieved": 0, "relevant_chunks": 0, "arc_params": {}, "contexts": []}
        a_lat = time.perf_counter() - t0
        a_prec = (a["relevant_chunks"] / a["chunks_retrieved"]) if a["chunks_retrieved"] else 0.0

        if mode == "llm":
            a_ok = await scorer(question, ground_truth, a["answer"])
        else:
            a_ok = scorer(ground_truth, a["answer"])

        a_correct.append(a_ok)
        a_latency.append(a_lat)
        a_precision.append(a_prec)
        a_hallucination.append(a["hallucination_flagged"])

        logger.info(
            f"  Baseline: {'✓' if b_ok else '✗'} latency={b_lat:.2f}s precision={b_prec:.2f}"
        )
        logger.info(
            f"  ACARA:    {'✓' if a_ok else '✗'} latency={a_lat:.2f}s precision={a_prec:.2f} "
            f"halluc={a['hallucination_flagged']}"
        )

        per_question_log.append({
            "id": i,
            "question": question,
            "ground_truth": ground_truth,
            "baseline": {
                "answer": b["answer"][:300],
                "correct": b_ok,
                "latency_s": round(b_lat, 3),
                "retrieval_precision": round(b_prec, 3),
            },
            "acara": {
                "answer": a["answer"][:300],
                "correct": a_ok,
                "latency_s": round(a_lat, 3),
                "retrieval_precision": round(a_prec, 3),
                "hallucination_flagged": a["hallucination_flagged"],
                "arc_params": a.get("arc_params", {}),
            },
        })

    # ─────────────────────────────────────────────────────────────────────────
    # Aggregate
    # ─────────────────────────────────────────────────────────────────────────
    b_acc  = mean(b_correct)
    a_acc  = mean(a_correct)
    b_lat  = mean(b_latency)
    a_lat  = mean(a_latency)
    b_prec = mean(b_precision)
    a_prec = mean(a_precision)
    a_hall = mean(a_hallucination)

    summary = {
        "eval_timestamp": datetime.utcnow().isoformat() + "Z",
        "judge_mode": mode,
        "n_questions": total,
        "metrics": {
            "answer_accuracy": {
                "baseline": round(b_acc, 4),
                "acara":    round(a_acc, 4),
                "delta":    round(a_acc - b_acc, 4),
            },
            "hallucination_rate": {
                "baseline": "N/A (no validator)",
                "acara":    round(a_hall, 4),
            },
            "retrieval_precision": {
                "baseline": round(b_prec, 4),
                "acara":    round(a_prec, 4),
                "delta":    round(a_prec - b_prec, 4),
            },
            "avg_latency_seconds": {
                "baseline": round(b_lat, 3),
                "acara":    round(a_lat, 3),
                "delta":    round(a_lat - b_lat, 3),
            },
        },
        "per_question": per_question_log,
    }

    # ─────────────────────────────────────────────────────────────────────────
    # Print Report
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  ACARA EVALUATION REPORT")
    print("=" * 80)
    print(f"  Questions evaluated : {total}")
    print(f"  Judge mode          : {mode}")
    print(f"  Timestamp           : {summary['eval_timestamp']}")
    print("-" * 80)
    print(f"  {'Metric':<30} {'Baseline':>12} {'ACARA':>12} {'Δ':>12}")
    print("-" * 80)
    print(f"  {'Answer Accuracy':<30} {b_acc:>12.1%} {a_acc:>12.1%} {a_acc-b_acc:>+12.1%}")
    print(f"  {'Hallucination Rate':<30} {'N/A':>12} {a_hall:>12.1%}")
    print(f"  {'Retrieval Precision':<30} {b_prec:>12.1%} {a_prec:>12.1%} {a_prec-b_prec:>+12.1%}")
    print(f"  {'Avg Latency (s)':<30} {b_lat:>12.2f} {a_lat:>12.2f} {a_lat-b_lat:>+12.2f}")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────────────────
    # Save JSON results
    # ─────────────────────────────────────────────────────────────────────────
    out_file = output_path or f"eval_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Results saved → {out_file}")
    print(f"  📄 Full results saved → {out_file}\n")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# CLI Entry Point
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(path: str, n: int) -> list[dict]:
    """Load JSONL where each line has 'question' and 'answer' keys."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Support MedQA-style keys
            if "question" not in obj and "Question" in obj:
                obj["question"] = obj.pop("Question")
            if "answer" not in obj and "Answer" in obj:
                obj["answer"] = obj.pop("Answer")
            items.append(obj)
            if len(items) >= n:
                break
    return items


def main():
    parser = argparse.ArgumentParser(description="ACARA Evaluation Benchmark")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Path to JSONL test set (question/answer pairs). "
                             "Defaults to built-in 500-question set.")
    parser.add_argument("--n", type=int, default=500,
                        help="Max questions to evaluate (default: 500)")
    parser.add_argument("--mode", choices=["llm", "exact"], default="llm",
                        help="Scoring mode: llm (GPT-4o-mini judge) or exact (keyword match)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    args = parser.parse_args()

    if args.dataset:
        test_set = load_dataset(args.dataset, args.n)
        logger.info(f"Loaded {len(test_set)} questions from {args.dataset}")
    else:
        test_set = SAMPLE_TEST_SET[: args.n]
        logger.info(f"Using built-in sample set ({len(test_set)} questions)")

    asyncio.run(run_eval(test_set, mode=args.mode, output_path=args.output))


if __name__ == "__main__":
    main()
