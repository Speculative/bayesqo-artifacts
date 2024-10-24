import asyncio
import time
from dataclasses import dataclass
from pprint import pformat
from typing import Optional

import boto3  # type: ignore
import psycopg  # type: ignore
from constants import USE_LOGGER

if USE_LOGGER:
    from logger.log import *
    from loguru import logger as l

ACCESS_KEY_ID = "<REMOVED FOR ANONYMIZATION>"
ACCESS_KEY_SECRET = "<REMOVED FOR ANONYMIZATION>"

IMDB_DB_USER = "<REMOVED FOR ANONYMIZATION>"
IMDB_DB_PASSWORD = "<REMOVED FOR ANONYMIZATION>"

LAUNCH_TEMPLATE_IDS = {
    "us-east-2": "<REMOVED FOR ANONYMIZATION>",
    "us-west-2": "<REMOVED FOR ANONYMIZATION>",
}

POSTGRES_PORT = 9940


@dataclass
class ExecutionEnvironment:
    host: str
    port: int
    user: str = IMDB_DB_USER
    password: str = IMDB_DB_PASSWORD


def _get_client(region_name: str):
    if region_name not in LAUNCH_TEMPLATE_IDS:
        raise ValueError(f"Region {region_name} not supported")

    client = boto3.client(
        "ec2",
        region_name=region_name,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_KEY_SECRET,
    )
    ec2 = boto3.resource(
        "ec2",
        region_name=region_name,
        aws_access_key_id=ACCESS_KEY_ID,
        aws_secret_access_key=ACCESS_KEY_SECRET,
    )
    return client, ec2


def find_instances(
    *,
    ids: Optional[list[str]] = None,
    run_name: Optional[str] = None,
    region_name: str = "us-east-2",
) -> list[tuple[str, str, str]]:
    """Returns a list of running/pending instances with their id, public ip address, and state."""

    if ids is not None and run_name is not None:
        raise ValueError("Cannot find running instances by both ids and run_name")

    client, _ = _get_client(region_name)

    filters = [
        {
            "Name": "instance-state-name",
            "Values": ["pending", "running"],
        }
    ]
    if ids is not None:
        filters.append(
            {
                "Name": "instance-id",
                "Values": ids,
            }
        )
    elif run_name is not None:
        filters.append(
            {
                "Name": "tag:run_name",
                "Values": [run_name],
            }
        )
    response = client.describe_instances(Filters=filters)
    if len(response["Reservations"]) == 0:
        # No instances
        return []
    else:
        all_instances = []
        reservations = response["Reservations"]
        for reservation in reservations:
            reservation_instances = reservation["Instances"]
            all_instances += [
                (
                    instance["InstanceId"],
                    instance["PublicIpAddress"],
                    instance["State"]["Name"],
                )
                for instance in reservation_instances
            ]
        return all_instances


async def _analyze(exec_env: ExecutionEnvironment):
    # Give Postgres a chance to start
    if USE_LOGGER:
        l.info("Waiting 4 minutes to give table restore time to complete")
    await asyncio.sleep(4 * 60)

    while True:
        try:
            if USE_LOGGER:
                l.info(f"Waiting for ANALYZE on {exec_env.host}")
            start_time = time.time()
            async with await psycopg.AsyncConnection.connect(
                host=exec_env.host,
                port=exec_env.port,
                dbname="imdb",
                user=IMDB_DB_USER,
                password=IMDB_DB_PASSWORD,
                # ANALYZE has to be run outside of a transaction
                autocommit=True,
            ) as aconn:
                async with aconn.cursor() as cur:
                    await cur.execute("ANALYZE")
                    finish_time = time.time()
                    if USE_LOGGER:
                        l.info(
                            f"Finished ANALYZE on {exec_env.host} after {finish_time - start_time:.2f} seconds"
                        )
                    return
        except psycopg.OperationalError as e:
            # Postgres is probably still starting
            if USE_LOGGER:
                l.info(
                    f"Failed to connect to {exec_env.host}: {e}. Retrying in 30 seconds..."
                )
            await asyncio.sleep(30)


async def _analyze_all(exec_envs: list[ExecutionEnvironment]):
    analyze_tasks = [asyncio.create_task(_analyze(exec_env)) for exec_env in exec_envs]
    await asyncio.gather(*analyze_tasks)


def analyze_all(exec_envs: list[ExecutionEnvironment]):
    asyncio.run(_analyze_all(exec_envs))


def provision_instances(run_name: str, num: int, region_name: str = "us-east-2"):
    instances = find_instances(run_name=run_name, region_name=region_name)
    if len(instances) > 0:
        raise RuntimeError("Instances already running, cowardly refusing to provision")

    start_time = time.time()
    _, ec2 = _get_client(region_name)
    if USE_LOGGER:
        l.info(f"Launching {num} new instances for run {run_name}")
    created_instances = ec2.create_instances(
        MinCount=num,
        MaxCount=num,
        LaunchTemplate={"LaunchTemplateId": LAUNCH_TEMPLATE_IDS[region_name]},
        TagSpecifications=[
            {
                "ResourceType": "instance",
                "Tags": [{"Key": "run_name", "Value": run_name}],
            }
        ],
    )

    formatted_instances = pformat(
        [
            (instance.id, instance.public_ip_address, instance.state["Name"])
            for instance in created_instances
        ]
    )
    if USE_LOGGER:
        l.info(
            f"Launched: {formatted_instances}",
        )

    waiting_instance_ids = [instance.id for instance in created_instances]

    waiting = True
    while waiting:
        time.sleep(5)
        pending_instances = find_instances(
            ids=waiting_instance_ids, region_name=region_name
        )
        if all(status == "running" for _, _, status in pending_instances):
            waiting = False

    launched_time = time.time()
    if USE_LOGGER:
        l.info(f"Launched all instances in {launched_time - start_time:.2f} seconds")

    exec_envs = [
        ExecutionEnvironment(host, POSTGRES_PORT) for _, host, _ in pending_instances
    ]

    analyze_all(exec_envs)

    analyze_finish_time = time.time()
    if USE_LOGGER:
        l.info(
            f"Ran ANALYZE on all instances in {analyze_finish_time - launched_time:.2f} seconds",
        )

    return exec_envs


def deprovision_instances(
    run_name: Optional[str] = None, region_name: str = "us-east-2"
):
    start_time = time.time()
    existing_instances = find_instances(run_name=run_name, region_name=region_name)

    if len(existing_instances) == 0:
        if run_name is not None:
            if USE_LOGGER:
                l.warning(f"No instances to terminate for run {run_name}")
        else:
            if USE_LOGGER:
                l.warning("No instances to terminate across all runs")
        return

    client, _ = _get_client(region_name)
    if run_name is not None:
        if USE_LOGGER:
            l.info(
                f"Terminating {len(existing_instances)} instances from run {run_name}"
            )
    else:
        if USE_LOGGER:
            l.info(f"Terminating ALL {len(existing_instances)} instances")
    client.terminate_instances(
        InstanceIds=[instance_id for instance_id, _, _ in existing_instances]
    )
    waiting = True
    while waiting:
        time.sleep(5)
        pending_instances = find_instances(run_name=run_name, region_name=region_name)
        if len(pending_instances) == 0:
            waiting = False

    finish_time = time.time()
    if USE_LOGGER:
        l.info(f"All instances terminated in {finish_time - start_time:.2f} seconds")


def retrieve_instances(run_name: str, region_name: str = "us-east-2"):
    instances = find_instances(run_name=run_name, region_name=region_name)
    return [ExecutionEnvironment(host, POSTGRES_PORT) for _, host, _ in instances]


if __name__ == "__main__":
    region_name = "us-east-2"
    existing_instances = find_instances(region_name=region_name)
    if USE_LOGGER:
        l.info(f"There are {len(existing_instances)} instances in {region_name}")
