"""
This script will replace the RunAI scheduler. An example of running this is as follows, though you can also give a file with a list of tasks:

python parallel_launch.py --seed 0 --num_per_gpu 2 --task_ids JOB_1A,JOB_1B,JOB_1C,JOB_2A --wandb_entity <REMOVED FOR ANONYMIZATION> --init_w_bao True

This will detect the number of GPUs available and run 2 tasks per GPU. As soon as a task finishes a new one will be started.

So, for example, you could allocate an entire A5000 node then launch this with 5 runs per GPU and it will run 50 tasks in parallel.
"""

import multiprocessing as mp
import os
import signal
import sys
import time
import traceback

import fire
import torch

sys.path.append("../")

from lolbo_scripts.info_transformer_vae_optimization import (
    InfoTransformerVAEOptimization,
)


def isolated_run_task(task_id: str, gpu_id: int, *args: list, **kwargs: dict) -> bool:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    runner = None
    try:
        with torch.cuda.device(gpu_id):
            runner = InfoTransformerVAEOptimization(workload_name=task_id, *args, **kwargs)
            runner.run_lolbo()
        return True  # Task completed successfully
    except Exception as e:
        if runner is not None:
            runner.done()

        print(f"Error in task {task_id}: {str(e)}")
        traceback.print_exc()
        raise e


def worker(
    task_queue: mp.JoinableQueue,
    gpu_id: int,
    runner: int,
    max_retries: int = 3,
    retry_delay: int = 60,
    *args: list,
    **kwargs: dict,
) -> None:
    while True:
        task_id = task_queue.get()
        print("#" * 80)
        if task_id is None:
            print(f"GPU {gpu_id} ({runner}) exiting")
            task_queue.task_done()
            break

        retries = 0
        while retries < max_retries:
            print(f"GPU {gpu_id} ({runner}) running task {task_id} (attempt {retries + 1})")
            p = mp.Process(target=isolated_run_task, args=(task_id, gpu_id, *args), kwargs=kwargs)
            p.start()
            p.join()

            if p.exitcode == 0:
                print(f"GPU {gpu_id} ({runner}) successfully completed task {task_id}")
                break
            else:
                retries += 1
                if retries < max_retries:
                    print(f"GPU {gpu_id} ({runner}) task {task_id} failed. Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"GPU {gpu_id} ({runner}) task {task_id} failed after {max_retries} attempts.")

        task_queue.task_done()
        print(f"GPU {gpu_id} ({runner}) finished processing task {task_id}")


def launch_tasks(
    num_per_gpu: int,
    task_ids: list[str] | str = None,
    task_file: str = None,
    *args: list,
    **kwargs: dict,
) -> None:
    """
    Launches tasks in parallel on multiple GPUs.

    Args:
        num_per_gpu (int): Number of tasks to run per GPU.
        task_ids (list[str]): List of task IDs (e.g. JOB_1A,JOB_1B,JOB_1C)
        task_file (str): Path to a file containing task IDs, one per line.
        *args (list): args to pass to optimization.
        **kwargs (dict): kwargs to pass to optimization.

    Raises:
        RuntimeError: If no GPUs are available.

    Returns:
        None
    """
    if (task_ids is None) == (task_file is None):
        raise ValueError("Exactly one of task_ids or task_file must be provided")

    if task_file is not None:
        with open(task_file, "r") as f:
            task_ids = f.read().splitlines()
    else:
        if isinstance(task_ids, str):
            task_ids = [task_ids]

    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available")

    task_queue = mp.JoinableQueue()
    processes = []

    def signal_handler(sig, frame):
        print("Interrupt received, terminating processes...")
        while not task_queue.empty():
            task_queue.get()
            task_queue.task_done()
        for _ in range(num_gpus):
            task_queue.put(None)
        for proc in processes:
            proc.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    for task_id in task_ids:
        task_queue.put(task_id)

    for gpu_id in range(num_gpus):
        for runner in range(num_per_gpu):
            proc = mp.Process(target=worker, args=(task_queue, gpu_id, runner, *args), kwargs=kwargs)
            proc.start()
            processes.append(proc)
            time.sleep(1)

    # Stop processes when all tasks are done
    for _ in range(num_gpus * num_per_gpu):
        task_queue.put(None)

    task_queue.join()

    for proc in processes:
        proc.join()


if __name__ == "__main__":
    fire.Fire(launch_tasks)
