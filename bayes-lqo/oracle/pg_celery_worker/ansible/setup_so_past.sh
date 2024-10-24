#!/usr/bin/env bash

if [ "$( sudo -u postgres psql -XtAc "SELECT 1 FROM pg_database WHERE datname='so_past'" )" = '1' ]; then
  # Database already exists
  echo "Skipping so_past setup"
else
  # configure & load so_past
  su postgres -c 'createuser -s so'
  su postgres -c 'createdb so_past'
  psql -U so -h localhost -c 'select 1;'
  pg_restore -U so -h localhost -d so_past -c /root/so_past || true
fi

# This script is always safe to run
psql -U so -h localhost -d so_past -f /root/configure_so_indexes.sql
