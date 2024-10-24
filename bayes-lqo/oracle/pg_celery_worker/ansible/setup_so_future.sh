#!/usr/bin/env bash

if [ "$( sudo -u postgres psql -XtAc "SELECT 1 FROM pg_database WHERE datname='so_future'" )" = '1' ]; then
  # Database already exists
  echo "Skipping so_future setup"
elif [ "$( sudo -u postgres psql -XtAc "SELECT 1 FROM pg_database WHERE datname='so'" )" = '1' ]; then
  # Rename so to so_future
  sudo -u postgres psql -c "ALTER DATABASE so RENAME TO so_future"
else
  # configure & load so_future
  # The so_pg13 dump assumes the database is named "so", so we have to start with that name
  su postgres -c 'createuser -s so'
  su postgres -c 'createdb so'
  psql -U so -h localhost -c 'select 1;'
  pg_restore -U so -h localhost -d so -c /root/so_pg13 || true
  # Then rename it to so_future after the restore
  sudo -u postgres psql -c "ALTER DATABASE so RENAME TO so_future"
fi

# This script is always safe to run
psql -U so -h localhost -d so_future -f /root/configure_so_indexes.sql
