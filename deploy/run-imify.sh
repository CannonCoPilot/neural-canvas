#!/bin/bash
# Imify API for neural-canvas (the free reverse-image arm): loopback only, live search only —
# no background tracker, no crawler, no face search. Source + venv + secrets (local.env, SearXNG
# secret and admin key) live in ~/Claude/Staging/verified/imify; the reviewed commit is b80f58a.
D="$HOME/Claude/Staging/verified/imify"
set -a; . "$D/local.env"; set +a
export STORAGE_BACKEND=sqlite DATA_DIR="$D/data" SEARXNG_URL=http://127.0.0.1:8888 \
  AUTH_MODE=open AUTO_TRACK_SEARCHES=false INDEXER_ENABLED=false FACEBOOK_RESOLUTION_ENABLED=false \
  REVERSE_PROVIDERS=searxng,bing,yandex GOOGLE_UPLOAD_ENABLED=false LOG_LEVEL=INFO
mkdir -p "$DATA_DIR" "$D/logs"
cd "$D/src" && exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8890 --workers 1
