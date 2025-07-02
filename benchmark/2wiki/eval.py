import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List

from hirag_prod import HiRAG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from dotenv import load_dotenv

load_dotenv("/chatbot/.env")


def calculate_metrics(
    retrieved_chunks: List[str],
    supporting_facts: List[List[str]],
    k_values: List[int] = [1, 3, 5, 10],
) -> Dict[str, float]:
    """
    Calculate retrieval metrics: Recall@k, Precision@k, and MRR

    Args:
        retrieved_chunks: List of retrieved chunk texts
        supporting_facts: List of [title, sentence_id] pairs from ground truth
        k_values: List of k values to calculate metrics for

    Returns:
        Dictionary with metric scores
    """
    metrics = {}

    # Extract supporting fact strings (assuming they are titles or key phrases)
    supporting_strings = [fact[0] for fact in supporting_facts]

    # Find which retrieved chunks contain supporting facts
    relevant_positions = []
    for i, chunk in enumerate(retrieved_chunks):
        for supporting_string in supporting_strings:
            if supporting_string.lower() in chunk.lower():
                relevant_positions.append(i)
                break

    # Calculate metrics for different k values
    for k in k_values:
        if k <= len(retrieved_chunks):
            # Recall@k: proportion of relevant items retrieved in top-k
            relevant_in_k = sum(1 for pos in relevant_positions if pos < k)
            recall_k = (
                relevant_in_k / len(supporting_strings) if supporting_strings else 0
            )

            # Precision@k: proportion of retrieved items that are relevant in top-k
            precision_k = relevant_in_k / k if k > 0 else 0

            metrics[f"recall@{k}"] = recall_k
            metrics[f"precision@{k}"] = precision_k

    # Calculate MRR (Mean Reciprocal Rank)
    reciprocal_ranks = []
    for supporting_string in supporting_strings:
        for i, chunk in enumerate(retrieved_chunks):
            if supporting_string.lower() in chunk.lower():
                reciprocal_ranks.append(1.0 / (i + 1))
                break
        else:
            reciprocal_ranks.append(0.0)

    mrr = statistics.mean(reciprocal_ranks) if reciprocal_ranks else 0
    metrics["mrr"] = mrr

    return metrics


async def evaluate_single_question(
    index: HiRAG, question_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate a single question and return metrics and timing info
    """
    question = question_data["question"]
    supporting_facts = question_data["supporting_facts"]

    # Measure retrieval time
    start_time = time.time()

    try:
        # Perform retrieval
        results = await index.query_all(question)

        retrieval_time = time.time() - start_time

        # Extract chunk texts (adjust based on your actual data structure)
        retrieved_chunks = []
        if "chunks" in results and results["chunks"]:
            # Assuming chunks is a list of dictionaries with text content
            for chunk in results["chunks"]:
                text = chunk.get("text")
                retrieved_chunks.append(text)

        # Calculate metrics
        metrics = calculate_metrics(retrieved_chunks, supporting_facts)

        return {
            "question": question,
            "retrieval_time": retrieval_time,
            "metrics": metrics,
            "num_retrieved": len(retrieved_chunks),
            "supporting_facts_count": len(supporting_facts),
            "success": True,
        }

    except Exception as e:
        logger.error(f"Error processing question: {question}. Error: {str(e)}")
        return {
            "question": question,
            "retrieval_time": time.time() - start_time,
            "metrics": {},
            "num_retrieved": 0,
            "supporting_facts_count": len(supporting_facts),
            "success": False,
            "error": str(e),
        }


async def evaluate(
    benchmark_file: str, n_questions: int = 100, max_concurrent: int = 5
):
    """
    Evaluate retrieval performance on benchmark questions

    Args:
        benchmark_file: Path to JSON file containing benchmark questions
        n_questions: Number of questions to evaluate (from the beginning)
        max_concurrent: Maximum number of concurrent requests
    """
    # Initialize HiRAG index
    index = await HiRAG.create()

    # Index the document
    document_path = f"benchmark/2wiki/2wiki_subcorpus.txt"
    content_type = "text/plain"
    document_meta = {
        "type": "txt",
        "filename": "2wiki_subcorpus.txt",
        "uri": document_path,
        "private": False,
    }

    logger.info("Indexing document...")
    await index.insert_to_kb(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
    )
    logger.info("Document indexed successfully")

    # Load benchmark questions
    with open(benchmark_file, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    # Limit to first n questions
    questions_data = questions_data[:n_questions]
    logger.info(f"Evaluating {len(questions_data)} questions")

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)

    async def evaluate_with_semaphore(question_data):
        async with semaphore:
            return await evaluate_single_question(index, question_data)

    # Process all questions concurrently
    start_time = time.time()
    results = await asyncio.gather(
        *[evaluate_with_semaphore(q_data) for q_data in questions_data]
    )
    total_time = time.time() - start_time

    # Aggregate results
    successful_results = [r for r in results if r["success"]]
    failed_count = len(results) - len(successful_results)

    if not successful_results:
        logger.error("No successful retrievals to analyze")
        return

    # Calculate aggregate metrics
    aggregate_metrics = defaultdict(list)
    retrieval_times = []

    for result in successful_results:
        retrieval_times.append(result["retrieval_time"])
        for metric, value in result["metrics"].items():
            aggregate_metrics[metric].append(value)

    # Calculate mean metrics
    mean_metrics = {}
    for metric, values in aggregate_metrics.items():
        mean_metrics[f"mean_{metric}"] = statistics.mean(values)
        mean_metrics[f"std_{metric}"] = (
            statistics.stdev(values) if len(values) > 1 else 0
        )

    # Print results
    print("\n" + "=" * 80)
    print("BENCHMARK EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nDataset Statistics:")
    print(f"  Total questions: {len(questions_data)}")
    print(f"  Successful retrievals: {len(successful_results)}")
    print(f"  Failed retrievals: {failed_count}")

    print(f"\nTiming Statistics:")
    print(f"  Total evaluation time: {total_time:.2f} seconds")
    print(f"  Average retrieval time: {statistics.mean(retrieval_times):.3f} seconds")
    print(f"  Median retrieval time: {statistics.median(retrieval_times):.3f} seconds")
    print(f"  Min retrieval time: {min(retrieval_times):.3f} seconds")
    print(f"  Max retrieval time: {max(retrieval_times):.3f} seconds")

    print(f"\nRetrieval Metrics:")
    k_values = [1, 3, 5, 10]
    for k in k_values:
        if f"mean_recall@{k}" in mean_metrics:
            print(
                f"  Recall@{k}: {mean_metrics[f'mean_recall@{k}']:.3f} (±{mean_metrics[f'std_recall@{k}']:.3f})"
            )
            print(
                f"  Precision@{k}: {mean_metrics[f'mean_precision@{k}']:.3f} (±{mean_metrics[f'std_precision@{k}']:.3f})"
            )

    if "mean_mrr" in mean_metrics:
        print(f"  MRR: {mean_metrics['mean_mrr']:.3f} (±{mean_metrics['std_mrr']:.3f})")

    # Save detailed results to file
    output_file = f"evaluation_results_{n_questions}q_{int(time.time())}.json"
    detailed_results = {
        "summary": {
            "total_questions": len(questions_data),
            "successful_retrievals": len(successful_results),
            "failed_retrievals": failed_count,
            "total_time": total_time,
            "mean_metrics": mean_metrics,
            "timing_stats": {
                "mean": statistics.mean(retrieval_times),
                "median": statistics.median(retrieval_times),
                "min": min(retrieval_times),
                "max": max(retrieval_times),
            },
        },
        "individual_results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 80)


if __name__ == "__main__":
    # Run evaluation
    # You can adjust these parameters:
    # - benchmark_file: path to your JSON benchmark file
    # - n_questions: number of questions to evaluate
    # - max_concurrent: maximum concurrent requests to avoid overwhelming the system
    asyncio.run(
        evaluate(
            benchmark_file="benchmark/2wiki/2wikimultihopqa_clean.json",
            n_questions=10,
            max_concurrent=25,
        )
    )
