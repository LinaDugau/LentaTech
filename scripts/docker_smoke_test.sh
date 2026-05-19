#!/usr/bin/env bash
# Docker end-to-end smoke test.
# Запускает compose, ждёт healthcheck, отправляет видео через /upload и
# проверяет что job попадает в очередь. Не дожидается окончания pipeline
# (50+ минут — заведомо медленно для smoke-теста).
#
# Usage:  bash scripts/docker_smoke_test.sh
set -uo pipefail

VIDEO="${VIDEO:-Данные/43_15/43_15.mp4}"  # 22M — самое маленькое
TIMEOUT_HEALTH=180
RC=0

step() { printf "\n\033[1;36m== %s ==\033[0m\n" "$*"; }
ok()   { printf "  \033[32m✓\033[0m %s\n" "$*"; }
fail() { printf "  \033[31m✗\033[0m %s\n" "$*"; RC=1; }

step "1) docker compose up -d"
docker compose up -d 2>&1 | tail -10 || { fail "compose up failed"; exit 1; }

step "2) wait for API health (timeout ${TIMEOUT_HEALTH}s)"
deadline=$(( $(date +%s) + TIMEOUT_HEALTH ))
while [ $(date +%s) -lt "$deadline" ]; do
    if curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
        ok "API healthy"
        curl -s http://localhost:8000/health | head -c 400; echo
        break
    fi
    sleep 3
done
if ! curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
    fail "API didn't become healthy within ${TIMEOUT_HEALTH}s"
    docker compose logs --tail=40 api
    exit 1
fi

step "3) GET / (endpoint list)"
curl -s http://localhost:8000/ | head -c 400; echo

step "4) POST /upload (test video)"
if [ ! -f "$VIDEO" ]; then
    fail "test video missing: $VIDEO"; exit 1
fi
RESP=$(curl -s -F "file=@${VIDEO}" http://localhost:8000/upload)
echo "  resp: $RESP"
JOB=$(printf "%s" "$RESP" | python3 -c "import sys,json; print(json.load(sys.stdin).get('job_id',''))" 2>/dev/null || true)
if [ -z "$JOB" ]; then
    fail "no job_id in response"; exit 1
fi
ok "job_id = $JOB"

step "5) GET /status/$JOB (3s sleep — даёт celery подхватить)"
sleep 3
curl -s "http://localhost:8000/status/$JOB" | head -c 400; echo

step "6) UI/dashboard health"
curl -fsS -o /dev/null http://localhost:8501 && ok "UI :8501 OK" || fail "UI :8501 down"
curl -fsS -o /dev/null http://localhost:8502 && ok "Dashboard :8502 OK" || fail "Dashboard :8502 down"

step "7) Prometheus metrics"
curl -s http://localhost:8000/metrics | head -3
curl -s http://localhost:9101/metrics 2>&1 | head -1

step "8) Container status"
docker compose ps

echo
if [ $RC -eq 0 ]; then
    printf "\033[1;32m=== SMOKE TEST PASSED ===\033[0m\n"
else
    printf "\033[1;31m=== SMOKE TEST FAILED ===\033[0m\n"
    docker compose logs --tail=60 api worker
fi
exit $RC
