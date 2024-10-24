DROP TABLE IF EXISTS job;
DROP TYPE IF EXISTS job_status;
CREATE TYPE job_status AS ENUM ('submitted', 'in-progress', 'complete');

CREATE TABLE job (
    id bigserial PRIMARY KEY,
    sql_statement text NOT NULL,
    target_db text NOT NULL,
    db_user text NOT NULL,
    timeout_ms int NOT NULL,
    issued_at TIMESTAMP DEFAULT current_timestamp NOT NULL,
    status job_status NOT NULL DEFAULT 'submitted',
    taken_by text,
    taken_at TIMESTAMP,
    result text,
    finished_at TIMESTAMP
);