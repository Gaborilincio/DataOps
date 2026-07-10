#!/bin/bash
set -e

mkdir -p /app/logs /app/data/raw /app/data/processed /app/output/eda /app/output/modelo
chown -R appuser:appuser /app/logs /app/data /app/output
 
exec su appuser -c "$*"