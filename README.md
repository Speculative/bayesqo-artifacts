# Usage

The function you probably care about is in `oracle/oracle.py`.

- `oracle_easy()`: Query 1A from the Join Order Benchmark
- `oracle_hard()`: Query 16b from the Join Order Benchmark

Oracle functions take 3 arguments:

```py
oracle(
    envs: list[ExecutionEnvironment],
    query_specs: list[QueryExecutionSpec],
    progress_callback: Callable[[QueryResult], dict[str, float]],
)
```

Progress callback is invoked with a `QueryResult` whenever any query completes and can return a dictionary of new timeouts for other queries (only specified query ids will have their timeouts changed, an empty dict will change no timeouts).

# Organization

- `/bayes_lqo`
  - `codec/codec.py`: Query plan language encoder/decoder (query plan -> string, string -> query plan)
  - `workload`
    - `ceb-3k/`: CEB workload
    - `job/`: JOB workload
      - `schema.sql`: Schema for JOB and CEB workloads
    - `stack/`: Stack Overflow workload
      - `schema.sql`: Schema for Stack Overflow workload
    - `/workloads.py`: Load workloads from query SQL, create schema/query join graphs
  - `oracle/`: Code for the black box oracle & DB cluster workers
    - `oracle.py`: Entry point for evaluating black box oracle function on the DB cluster
    - `pg_celery_worker/`:
      - `pg_worker/`: Code for the DB worker service that executes queries in the queue
      - `ansible/`: DB cluster node configuration files
        - `postgresql.conf`: PostreSQL configuration used on DB cluster nodes
        - `setup_pg.sh`: Installs PostgreSQL, loads the IMDB dataset
        - `setup_so_future.sh`: Loads the Stack (future, 2019) dataset
        - `setup_so_past.sh`: Loads the Stack (past, 2017) dataset
        - `configure_so_indexes.sql`: Creates indexes for the Stack dataset (both past and future)
  - `training_data/`: Scripts to generate training data for the VAE
    - `gen_alias_workload.py`: Generates random queries
    - `planner.py`: Plans queries (with varying hint sets) using PostgreSQL
    - `codec.py`: Encode generated plans as strings
  - `db_eval`: Scripts to run baselines, generate plots
    - `random_plans.py`: Code for the "Random" baseline
    - `bao.py`: Code for the "Bao" baseline

# Testing Locally

Instead of provisioning EC2 instances, you can develop locally using a Docker image:

1. Be in the imdb_postgres directory:

   ```sh
   cd imdb_postgres
   ```

2. Download the imdb_pg11 dataset:

   ```sh
   wget -O imdb_pg11 --progress=dot:giga "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/2QYZBT/TGYUNU"
   ```

3. Build & run the docker image:

   ```sh
   docker build -t imdb-postgres:latest .

   docker run imdb-postgres
   ```

4. Get the container IP:

   ```sh
   docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container_name
   ```

5. Use this as your envs:

   ```py
   envs = [ExecutionEnvironment(CONTAINTER_IP_ADDRESS, 5432, "imdb", "imdb") for _ in range(BATCH_SIZE)]
   ```

# How to Run Optimization

Set up a weights and biases (wandb) account so you can automatically track optimization progress and results

https://wandb.ai/home

Your account will be set up with a wandb entity name (like a short username). Pass that in as `--wandb_entity <NAME>`

> [!IMPORTANT]
> All commands must be run from within the `bayes_lqo/lolbo_scripts` folder.

## JOB / CEB

- JOB task IDs are defined in [tasks/job_tasks.txt](tasks/job_tasks.txt)
- CEB task IDs are defined in [tasks/ceb_tasks.txt](tasks/ceb_tasks.txt)

To run optimization on a JOB / CEB task, run the following command with any given task ID:

```Bash
python3 info_transformer_vae_optimization.py --workload_name <TASK_ID> --wandb_entity <NAME> - run_lolbo - done
```

## Stack Overflow (SO)

- SO task IDs are defined in [tasks/so_tasks.txt](tasks/so_tasks.txt)

To run SO Past:

```Bash
python3 info_transformer_vae_optimization.py --workload_name <TASK_ID> --wandb_entity <NAME> --so_future False - run_lolbo - done
```

To run SO Future:

```Bash
python3 info_transformer_vae_optimization.py --workload_name <TASK_ID> --wandb_entity <NAME> --so_future True --force_past_vae False - run_lolbo - done
```

To run SO Future with Past VAE:

```Bash
python3 info_transformer_vae_optimization.py --workload_name <TASK_ID> --wandb_entity <NAME> --so_future True --force_past_vae True - run_lolbo - done
```
