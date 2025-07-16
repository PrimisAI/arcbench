import json
import os
import logging
import time
from pathlib import Path

from baseline import ask_baseline_llm
from utils.eval import compute_machine_score
from workflow import create_workflow
from tqdm import tqdm


def run_baseline_experiments(input_file: str,
                             output_file: str,
                             log_file: str = "logs/baseline_experiments.log") -> None:
    """
    Run baseline LLM experiments on questions and save results.

    Args:
        input_file (str): Path to input JSONL file.
        output_file (str): Path to output JSONL file.
        log_file (str): Path to log file.
    """
    # Setup logging
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )

    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logging.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    # Load existing results to avoid repeat processing
    processed_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed_ids.add(result["id"])
                except Exception:
                    continue  # Ignore corrupted lines

    logging.info(f"Found {len(processed_ids)} already processed items")

    # Read all data
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    total_rows = len(data)
    processed_count = 0
    failed_count = 0
    start_time = time.time()

    with tqdm(total=total_rows, desc="Processing", unit="row") as progress_bar, \
            open(output_path, "a", encoding="utf-8") as fout:
        for idx, item in enumerate(data):
            if item["id"] in processed_ids:
                progress_bar.update(1)
                continue

            try:
                question = item["question"]
                actual_answer = item["answer"]

                # Retry mechanism for API calls
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        predicted_answer = ask_baseline_llm(question)
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            raise e
                        time.sleep(2**retry)

                machine_score = compute_machine_score(actual_answer, predicted_answer)
                result_record = {
                    "id": item["id"],
                    "question": question,
                    "actual_answer": actual_answer,
                    "predicted_answer": predicted_answer,
                    "machine_score": machine_score,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                fout.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                fout.flush()
                processed_count += 1

            except Exception as e:
                failed_count += 1
                logging.error(f"Error processing item {item.get('id', idx)}: {str(e)}")
                failed_file = output_path.parent / "failed_items.jsonl"
                with open(failed_file, "a", encoding="utf-8") as f_failed:
                    failure_record = {
                        "id": item.get('id', idx),
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    f_failed.write(json.dumps(failure_record) + "\n")

            progress_bar.update(1)

    end_time = time.time()
    duration = end_time - start_time
    avg_time_per_item = duration / total_rows if total_rows > 0 else 0.0

    logging.info("Processing completed:")
    logging.info(f"Total items: {total_rows}")
    logging.info(f"Successfully processed: {processed_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info(f"Total time: {duration:.2f} seconds")
    logging.info(f"Average time per item: {avg_time_per_item:.2f} seconds")


def run_workflow_experiments(input_file: str,
                             output_file: str,
                             log_file: str = "logs/workflow_experiments.log",
                             workflow_id: str = "workflow") -> None:
    """
    Run agent workflow experiments on questions and save results.

    Args:
        input_file (str): Path to input JSONL file.
        output_file (str): Path to output JSONL file.
        log_file (str): Path to log file.
        workflow_id (str): Workflow id for storing conversation logs.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file)],
    )

    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logging.error(f"Input file not found: {input_file}")
        raise FileNotFoundError(f"Input file not found: {input_file}")

    logging.info(f"Loading data from {input_file}")
    data = []
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    total_rows = len(data)
    logging.info(f"Loaded {total_rows} items from input file")

    processed_ids = set()
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    result = json.loads(line)
                    processed_ids.add(result.get("id"))
                except Exception:
                    continue

    processed_count = len(processed_ids)
    logging.info(f"Found {processed_count} already processed items")
    start_time = time.time()
    failed_count = 0
    new_processed_count = 0

    with tqdm(total=total_rows, initial=processed_count, desc="Processing", unit="row") as progress_bar, \
            open(output_path, "a", encoding="utf-8") as fout:
        for item in data:
            if item["id"] in processed_ids:
                progress_bar.update(1)
                continue

            try:
                question = item["question"]
                actual_answer = item["answer"]
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        workflow = create_workflow(workflow_id=item["id"] + "_" + workflow_id)
                        predicted_answer = workflow.chat(question)
                        break
                    except Exception as e:
                        if retry == max_retries - 1:
                            raise e
                        logging.warning(f"Retry {retry + 1}/{max_retries} for item {item['id']}: {str(e)}")
                        time.sleep(2**retry)
                machine_score = compute_machine_score(actual_answer, predicted_answer)
                result_record = {
                    "id": item["id"],
                    "question": question,
                    "actual_answer": actual_answer,
                    "predicted_answer": predicted_answer,
                    "machine_score": machine_score,
                    "workflow_id": item["id"] + "_" + workflow_id,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                fout.write(json.dumps(result_record, ensure_ascii=False) + "\n")
                fout.flush()
                new_processed_count += 1

            except Exception as e:
                failed_count += 1
                logging.error(f"Error processing item {item.get('id')}: {str(e)}")
                failed_file = output_path.parent / "failed_workflow_items.jsonl"
                with open(failed_file, "a", encoding="utf-8") as f_failed:
                    failure_record = {
                        "id": item.get('id'),
                        "error": str(e),
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    f_failed.write(json.dumps(failure_record) + "\n")

            progress_bar.update(1)

    end_time = time.time()
    duration = end_time - start_time
    avg_time_per_item = duration / new_processed_count if new_processed_count > 0 else 0.0

    logging.info("Processing completed:")
    logging.info(f"Total items: {total_rows}")
    logging.info(f"Previously processed: {processed_count}")
    logging.info(f"Newly processed: {new_processed_count}")
    logging.info(f"Failed: {failed_count}")
    logging.info(f"Total time: {duration:.2f} seconds")
    logging.info(f"Average time per new item: {avg_time_per_item:.2f} seconds")


def recompute_scores(benchmark_file: str, result_file: str, output_file: str) -> None:
    """
    Recomputes the machine scores for predicted answers in result_file
    against actual answers in benchmark_file, and writes output_file.
    """
    # Load benchmark answers
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = [json.loads(line) for line in f]
    # Load predicted answers from previous results
    with open(result_file, "r", encoding="utf-8") as f:
        results = [json.loads(line) for line in f]
    with open(output_file, "w", encoding="utf-8") as fout:
        for result in results:
            idx = result["id"]
            predicted_answer = result["predicted_answer"]
            actual_answer = benchmark_data[int(idx)]["answer"]
            new_score = compute_machine_score(actual_answer, predicted_answer)
            updated_result = {
                "id": idx,
                "question": result["question"],
                "actual_answer": result["actual_answer"],
                "predicted_answer": result["predicted_answer"],
                "machine_score": new_score,
                "workflow_id": result["workflow_id"],
                "timestamp": result["timestamp"],
            }
            fout.write(json.dumps(updated_result, ensure_ascii=False) + "\n")
    print(f"Recomputed scores saved to '{output_file}'")


def print_model_info():
    gen_model = os.getenv("LLM", "unknown-model")
    eval_model = os.getenv("EVAL_LLM", "unknown-eval-model")
    print(f"Generation  LLM: {gen_model}")
    print(f"Evaluation LLM: {eval_model}\n")


if __name__ == "__main__":
    print_model_info()
    model_name = os.getenv("LLM", "unknown-model").replace("/", "_")

    # Baseline experiments
    run_baseline_experiments(input_file="data/ArcBench.jsonl", output_file=f"results/{model_name}_baseline.jsonl")

    # Workflow experiments
    run_workflow_experiments(input_file="data/ArcBench.jsonl",
                             output_file=f"results/{model_name}_workflow.jsonl",
                             workflow_id=model_name)
