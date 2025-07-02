import asyncio
import json
import logging
import statistics
import time
from collections import defaultdict
from typing import Any, Dict, List

from hirag_prod import HiRAG

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
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
    Same as the original eval.py for consistency
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


async def extract_chunk_ids_from_graph_retrieval(
    index: HiRAG, entities: List[Dict], neighbors: List[str], relations: List[str]
) -> List[str]:
    """
    Extract chunk IDs from retrieved entities and relations

    Args:
        index: HiRAG instance (for logging context)
        entities: Retrieved entities from query_entities
        neighbors: Neighbor entities from query_relations
        relations: Edge relations from query_relations

    Returns:
        List of unique chunk IDs associated with entities and relations
    """
    try:
        # Extract chunk IDs from entities
        entity_chunk_ids = set()
        for entity in entities:
            chunk_ids = entity.get("chunk_ids", [])
            if isinstance(chunk_ids, list):
                entity_chunk_ids.update(chunk_ids)
            elif isinstance(chunk_ids, str):
                entity_chunk_ids.add(chunk_ids)

        logger.debug(
            f"Found {len(entity_chunk_ids)} chunk IDs from {len(entities)} entities"
        )

        # Extract chunk IDs from neighbor entities
        neighbor_chunk_ids = set()
        for neighbor in neighbors:
            if hasattr(neighbor, "metadata") and hasattr(
                neighbor.metadata, "chunk_ids"
            ):
                chunk_ids = neighbor.metadata.chunk_ids
                if isinstance(chunk_ids, list):
                    neighbor_chunk_ids.update(chunk_ids)
                elif isinstance(chunk_ids, str):
                    neighbor_chunk_ids.add(chunk_ids)

        logger.debug(
            f"Found {len(neighbor_chunk_ids)} chunk IDs from {len(neighbors)} neighbor entities"
        )

        # Extract chunk IDs from relations
        relation_chunk_ids = set()
        for relation in relations:
            if hasattr(relation, "properties") and "chunk_id" in relation.properties:
                chunk_id = relation.properties["chunk_id"]
                if chunk_id:
                    relation_chunk_ids.add(chunk_id)

        logger.debug(
            f"Found {len(relation_chunk_ids)} chunk IDs from {len(relations)} relations"
        )

        # Combine all unique chunk IDs
        all_chunk_ids = entity_chunk_ids | neighbor_chunk_ids | relation_chunk_ids
        logger.debug(f"Total unique chunk IDs: {len(all_chunk_ids)}")

        return list(all_chunk_ids)

    except Exception as e:
        logger.error(f"Error extracting chunk IDs from graph retrieval: {e}")
        return []


async def rerank_chunks_with_query(
    index: HiRAG, chunk_ids: List[str], query: str, topn: int = 5
) -> List[str]:
    """
    Rerank retrieved chunks using the same VoyageAI reranker as the system
    Uses existing chunks in LanceDB, no need to recreate embeddings

    Args:
        index: HiRAG instance
        chunk_ids: List of chunk document_keys to rerank
        query: Original query for reranking
        topn: Number of top chunks to return after reranking

    Returns:
        List of reranked chunk texts (top N)
    """
    if not chunk_ids:
        return []

    if len(chunk_ids) <= topn:
        # If we have few chunks, get their texts and return them
        try:
            conditions = " OR ".join(
                [f"document_key == '{chunk_id}'" for chunk_id in chunk_ids]
            )
            results = await index.chunks_table.query().where(conditions).to_list()
            return [result["text"] for result in results]
        except Exception as e:
            logger.warning(f"Failed to get chunk texts: {e}")
            return []

    try:
        # Suppress Lance warnings during reranking
        lance_logger = logging.getLogger("lance")
        original_level = lance_logger.level
        lance_logger.setLevel(logging.ERROR)

        try:
            # Create query filter to only consider the retrieved chunks
            conditions = " OR ".join(
                [f"document_key == '{chunk_id}'" for chunk_id in chunk_ids]
            )

            # Get query embedding for vector search
            query_embeddings = await index.embedding_service.create_embeddings([query])
            query_embedding = query_embeddings[0].tolist()

            # Create vector query on existing chunks table
            vector_query = (
                index.chunks_table.query()
                .nearest_to(query_embedding)
                .distance_type("cosine")
            )

            # Set nprobes to avoid warnings
            if hasattr(vector_query, "nprobes"):
                vector_query = vector_query.nprobes(20)

            # Filter to only the chunks we want to rerank
            vector_query = vector_query.where(conditions)

            # Select needed columns and limit
            vector_query = vector_query.select(["text", "document_key"]).limit(
                len(chunk_ids)
            )

            # Apply the same reranker as the system
            reranked_query = index.vdb.strategy_provider.rerank_chunk_query(
                vector_query, query, topn
            )

            # Get reranked results
            reranked_results = await reranked_query.to_list()

            # Extract texts in reranked order
            reranked_texts = [result["text"] for result in reranked_results]

            logger.debug(
                f"Successfully reranked {len(chunk_ids)} chunks to top {len(reranked_texts)} using VoyageAI reranker"
            )
            return reranked_texts[:topn]

        finally:
            # Restore original logging level
            lance_logger.setLevel(original_level)

    except Exception as e:
        logger.warning(
            f"VoyageAI reranking failed, falling back to text extraction: {e}"
        )
        # Fallback: just get the text of the chunks without reranking
        try:
            conditions = " OR ".join(
                [f"document_key == '{chunk_id}'" for chunk_id in chunk_ids[:topn]]
            )
            results = await index.chunks_table.query().where(conditions).to_list()
            return [result["text"] for result in results]
        except Exception as fallback_e:
            logger.warning(f"Fallback also failed: {fallback_e}")
            return []


async def evaluate_single_question_graphrag(
    index: HiRAG, question_data: Dict[str, Any], question_idx: int, total_questions: int
) -> Dict[str, Any]:
    """
    Evaluate a single question using GraphRAG (entity + relation) retrieval

    Args:
        index: HiRAG instance
        question_data: Question and supporting facts
        question_idx: Current question index (for logging)
        total_questions: Total number of questions (for logging)

    Returns:
        Evaluation results dictionary
    """
    question = question_data["question"]
    supporting_facts = question_data["supporting_facts"]

    logger.info(
        f"[{question_idx + 1}/{total_questions}] Processing: {question[:100]}..."
    )

    # Measure retrieval time
    start_time = time.time()

    try:
        # Suppress LanceDB warnings during retrieval
        lance_logger = logging.getLogger("lance")
        original_level = lance_logger.level
        lance_logger.setLevel(logging.ERROR)

        try:
            # Perform entity and relation retrieval
            logger.debug(f"[{question_idx + 1}] Retrieving entities...")
            entities = await index.query_entities(question, topk=10, topn=5)
            logger.debug(f"[{question_idx + 1}] Retrieved {len(entities)} entities")

            logger.debug(f"[{question_idx + 1}] Retrieving relations...")
            neighbors, relations = await index.query_relations(
                question, topk=10, topn=5
            )
            logger.debug(
                f"[{question_idx + 1}] Retrieved {len(neighbors)} neighbors, {len(relations)} relations"
            )
        finally:
            # Restore original logging level
            lance_logger.setLevel(original_level)

        # Extract chunk IDs associated with entities and relations
        logger.debug(f"[{question_idx + 1}] Extracting associated chunk IDs...")
        chunk_ids = await extract_chunk_ids_from_graph_retrieval(
            index, entities, neighbors, relations
        )
        logger.debug(f"[{question_idx + 1}] Extracted {len(chunk_ids)} chunk IDs")

        # Rerank chunks using the same reranker as the system
        logger.debug(f"[{question_idx + 1}] Reranking chunks...")
        retrieved_chunks = await rerank_chunks_with_query(
            index, chunk_ids, question, topn=5
        )

        retrieval_time = time.time() - start_time
        logger.debug(
            f"[{question_idx + 1}] Retrieved {len(retrieved_chunks)} reranked chunks in {retrieval_time:.3f}s"
        )

        # Calculate metrics
        metrics = calculate_metrics(retrieved_chunks, supporting_facts)

        result = {
            "question": question,
            "retrieval_time": retrieval_time,
            "metrics": metrics,
            "num_chunk_ids": len(chunk_ids),
            "num_retrieved_chunks": len(retrieved_chunks),
            "num_entities": len(entities),
            "num_neighbors": len(neighbors),
            "num_relations": len(relations),
            "supporting_facts_count": len(supporting_facts),
            "success": True,
        }

        # Log metrics for this question
        recall_at_5 = metrics.get("recall@5", 0)
        precision_at_5 = metrics.get("precision@5", 0)
        mrr = metrics.get("mrr", 0)
        logger.info(
            f"[{question_idx + 1}] Metrics - R@5: {recall_at_5:.3f}, P@5: {precision_at_5:.3f}, MRR: {mrr:.3f}"
        )

        return result

    except Exception as e:
        retrieval_time = time.time() - start_time
        logger.error(f"[{question_idx + 1}] Error processing question: {e}")
        return {
            "question": question,
            "retrieval_time": retrieval_time,
            "metrics": {},
            "num_chunk_ids": 0,
            "num_retrieved_chunks": 0,
            "num_entities": 0,
            "num_neighbors": 0,
            "num_relations": 0,
            "supporting_facts_count": len(supporting_facts),
            "success": False,
            "error": str(e),
        }


async def evaluate_graphrag(
    benchmark_file: str, n_questions: int = 100, max_concurrent: int = 5
):
    """
    Evaluate GraphRAG retrieval performance on benchmark questions

    Args:
        benchmark_file: Path to JSON file containing benchmark questions
        n_questions: Number of questions to evaluate (from the beginning)
        max_concurrent: Maximum number of concurrent requests
    """
    logger.info("=" * 80)
    logger.info("STARTING GRAPHRAG EVALUATION")
    logger.info("=" * 80)
    logger.info(f"Configuration:")
    logger.info(f"  Benchmark file: {benchmark_file}")
    logger.info(f"  Questions to evaluate: {n_questions}")
    logger.info(f"  Max concurrent requests: {max_concurrent}")
    logger.info(f"  Evaluation type: Entity + Relation retrieval with reranking")

    # Initialize HiRAG index
    logger.info("Initializing HiRAG index...")
    index = await HiRAG.create()

    # Index the document
    document_path = f"benchmark/2wiki/2wiki_corpus.txt"
    content_type = "text/plain"
    document_meta = {
        "type": "txt",
        "filename": "2wiki_corpus.txt",
        "uri": document_path,
        "private": False,
    }

    logger.info("Indexing document corpus...")
    index_start_time = time.time()
    await index.insert_to_kb(
        document_path=document_path,
        content_type=content_type,
        document_meta=document_meta,
    )
    index_time = time.time() - index_start_time
    logger.info(f"Document indexed successfully in {index_time:.2f} seconds")

    # Load benchmark questions
    logger.info(f"Loading benchmark questions from {benchmark_file}...")
    with open(benchmark_file, "r", encoding="utf-8") as f:
        questions_data = json.load(f)

    # Limit to first n questions
    questions_data = questions_data[:n_questions]
    logger.info(f"Loaded {len(questions_data)} questions for evaluation")

    # Create semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Using max {max_concurrent} concurrent requests")

    async def evaluate_with_semaphore(question_data, idx):
        async with semaphore:
            return await evaluate_single_question_graphrag(
                index, question_data, idx, len(questions_data)
            )

    # Process all questions concurrently
    logger.info("Starting concurrent evaluation...")
    start_time = time.time()

    # Add progress logging for long evaluations
    async def log_progress():
        while True:
            await asyncio.sleep(60)  # Log every minute
            elapsed = time.time() - start_time
            logger.info(f"Evaluation in progress... {elapsed:.1f}s elapsed")

    # Start progress logging task
    progress_task = asyncio.create_task(log_progress())

    try:
        results = await asyncio.gather(
            *[
                evaluate_with_semaphore(q_data, idx)
                for idx, q_data in enumerate(questions_data)
            ]
        )
    finally:
        # Cancel progress logging
        progress_task.cancel()
        try:
            await progress_task
        except asyncio.CancelledError:
            pass

    total_time = time.time() - start_time
    logger.info(f"Evaluation completed in {total_time:.2f} seconds")

    # Aggregate results
    successful_results = [r for r in results if r["success"]]
    failed_count = len(results) - len(successful_results)

    if not successful_results:
        logger.error("No successful retrievals to analyze!")
        return

    logger.info(f"Successful evaluations: {len(successful_results)}/{len(results)}")
    if failed_count > 0:
        logger.warning(f"Failed evaluations: {failed_count}")

    # Calculate aggregate metrics
    aggregate_metrics = defaultdict(list)
    retrieval_times = []
    entity_counts = []
    neighbor_counts = []
    relation_counts = []
    chunk_id_counts = []
    chunk_counts = []

    for result in successful_results:
        retrieval_times.append(result["retrieval_time"])
        entity_counts.append(result["num_entities"])
        neighbor_counts.append(result["num_neighbors"])
        relation_counts.append(result["num_relations"])
        chunk_id_counts.append(result["num_chunk_ids"])
        chunk_counts.append(result["num_retrieved_chunks"])

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
    print("GRAPHRAG EVALUATION RESULTS")
    print("=" * 80)

    print(f"\nDataset Statistics:")
    print(f"  Total questions: {len(questions_data)}")
    print(f"  Successful retrievals: {len(successful_results)}")
    print(f"  Failed retrievals: {failed_count}")

    print(f"\nTiming Statistics:")
    print(f"  Total evaluation time: {total_time:.2f} seconds")
    print(f"  Average retrieval time: {statistics.mean(retrieval_times):.3f} seconds")
    print(f"  Median retrieval time: {statistics.median(retrieval_times):.3f} seconds")

    print(f"\nRetrieval Statistics:")
    print(f"  Average entities per query: {statistics.mean(entity_counts):.1f}")
    print(f"  Average neighbors per query: {statistics.mean(neighbor_counts):.1f}")
    print(f"  Average relations per query: {statistics.mean(relation_counts):.1f}")
    print(f"  Average chunk IDs per query: {statistics.mean(chunk_id_counts):.1f}")
    print(
        f"  Average final chunks per query (after rerank): {statistics.mean(chunk_counts):.1f}"
    )

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
    output_file = f"evaluation_results_graphrag_{n_questions}q_{int(time.time())}.json"
    detailed_results = {
        "evaluation_type": "graphrag",
        "summary": {
            "total_questions": len(questions_data),
            "successful_retrievals": len(successful_results),
            "failed_retrievals": failed_count,
            "total_time": total_time,
            "index_time": index_time,
            "mean_metrics": mean_metrics,
            "timing_stats": {
                "mean": statistics.mean(retrieval_times),
                "median": statistics.median(retrieval_times),
                "min": min(retrieval_times),
                "max": max(retrieval_times),
            },
            "retrieval_stats": {
                "mean_entities": statistics.mean(entity_counts),
                "mean_neighbors": statistics.mean(neighbor_counts),
                "mean_relations": statistics.mean(relation_counts),
                "mean_chunk_ids": statistics.mean(chunk_id_counts),
                "mean_final_chunks": statistics.mean(chunk_counts),
                "rerank_enabled": True,
            },
        },
        "individual_results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

    logger.info(f"Detailed results saved to: {output_file}")
    print(f"\nDetailed results saved to: {output_file}")
    print("=" * 80)

    # Clean up
    await index.clean_up()
    logger.info("Evaluation completed and resources cleaned up")


if __name__ == "__main__":
    # Run GraphRAG evaluation
    # Parameters:
    # - benchmark_file: path to your JSON benchmark file
    # - n_questions: number of questions to evaluate
    # - max_concurrent: maximum concurrent requests to avoid overwhelming the system

    try:
        asyncio.run(
            evaluate_graphrag(
                benchmark_file="benchmark/2wiki/2wikimultihopqa_clean.json",
                n_questions=1000,
                max_concurrent=25,
            )
        )
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
    except Exception as e:
        logger.error(f"Evaluation failed with error: {e}")
        raise
