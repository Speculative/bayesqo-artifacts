import json
import os
import time
from multiprocessing import Process

import psycopg

PG_PASS = "Qyom5*$G4UkHo1"


hostname = "unknown-host"
if os.path.exists("./hostname"):
    with open("./hostname") as f:
        hostname = f.read().strip()


def start_pg_force_restart(timeout_ms):
    assert timeout_ms >= 1, f"timeout must be at least 1ms (was {timeout_ms})"

    def runner():
        time.sleep(timeout_ms / 1000)
        print("Forcibly restarting the PG server...")
        os.system("systemctl restart postgresql")

    p = Process(target=runner)
    p.start()
    return p


GET_JOB_SQL = """
SELECT id, sql_statement, target_db, db_user, timeout_ms 
FROM job 
WHERE taken_by IS NULL 
      OR (taken_by = %s AND status != 'complete')
ORDER BY issued_at
LIMIT 1;
"""

print(f"{hostname} is waiting for a job...")
while True:
    with psycopg.connect(
        host="<REMOVED FOR ANONYMIZATION>",
        user="bayesopt",
        dbname="bayesopt",
        password=PG_PASS,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(GET_JOB_SQL, (hostname,))
            res = cur.fetchone()

            # if there are no jobs available, wait a while before checking again
            if res is None:
                time.sleep(5)
                continue

            # if we have a job, mark it as taken and commit
            j_id, sql, db, db_user, timeout_ms = res
            cur.execute(
                "UPDATE job SET taken_by = %s, taken_at=current_timestamp, status = 'in-progress' WHERE id = %s",
                (hostname, j_id),
            )
            conn.commit()

        print(f"Got job ID {j_id} to complete")
        result = {"worker": hostname}
        force_restart = None
        try:
            # first, we execute the query in cold cache, ignoring the result
            print(f"Prewarming for {2*timeout_ms + 10_000}ms")
            force_restart = start_pg_force_restart(2 * timeout_ms + 10_000)
            with psycopg.connect(dbname=db, user=db_user, host="localhost") as lconn:
                with lconn.cursor() as lcur:
                    lcur.execute(f"SET statement_timeout TO {2*timeout_ms}")
                    lcur.execute("SET client_encoding TO 'UTF8'")

                    # warm the cache
                    try:
                        lcur.execute(sql)
                        print("Prewarm worked, executing for real...")
                        result["prewarm"] = "complete"
                    except:  # noqa: E722
                        print("Prewarm failed, sleeping and executing for real...")
                        result["prewarm"] = "failed"
                        time.sleep(5)
            while force_restart.is_alive():
                force_restart.kill()
                time.sleep(0.5)

            # next, we execute the query and time it
            force_restart = start_pg_force_restart(2 * timeout_ms + 10_000)
            with psycopg.connect(dbname=db, user=db_user, host="localhost") as lconn:
                with lconn.cursor() as lcur:
                    lcur.execute(f"SET statement_timeout TO {timeout_ms}")
                    lcur.execute("SET client_encoding TO 'UTF8'")

                    start = time.time_ns()
                    lcur.execute(sql)
                    dur = time.time_ns() - start
                    result["result"] = lcur.fetchall()
                    result["status"] = "complete"
                    result["duration (ns)"] = dur
                    print(
                        f"Query completed, latency was {dur}ns (excluding fetchall())"
                    )

        except psycopg.errors.QueryCanceled as e:
            result["status"] = "timeout"
            result["message"] = str(e)
            result["duration (ns)"] = timeout_ms * 1000000
        except psycopg.errors.OperationalError as e:
            print("Operational error, bailing out:" + str(e))
            result = None
        except psycopg.errors.ProgrammingError as e:
            print("Possible query syntax error: " + str(e))
            result["status"] = "error"
            result["message"] = str(e)
        except AssertionError as e:
            print("Assertion failure in worker:" + str(e))
            result["status"] = "error"
            result["message"] = str(e)
        except Exception as e:
            print("Query failed: " + str(e))
            result["status"] = "failed"
            result["message"] = str(e)
            result["duration (ns)"] = timeout_ms * 1000000
        finally:
            # cancel the fallback PG restarter process
            if force_restart is not None:
                while force_restart.is_alive():
                    force_restart.kill()
                    time.sleep(0.5)

            # submit result to DB
            with conn.cursor() as cur:
                print(f"Finished job {j_id}, complete = {result is not None}")
                if result:
                    result = json.dumps(result)
                    cur.execute(
                        "UPDATE job SET status = 'complete', result = %s, finished_at = current_timestamp WHERE id = %s",
                        (result, j_id),
                    )
                    conn.commit()
                else:
                    print("Putting query back into queue and waiting...")
                    cur.execute(
                        "UPDATE job SET status = 'submitted', taken_by = NULL WHERE id = %s",
                        (j_id,),
                    )
                    conn.commit()
                    time.sleep(10)
