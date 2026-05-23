from __future__ import annotations

import shutil
from typing import Any, Dict

import redis

from app.config import (
    CELERY_BROKER_URL,
    DETECTOR_WEIGHTS_PATH,
    LLM_WEIGHTS_PATH,
    MIN_FREE_DISK_BYTES,
    TMP_DIR,
)


def build_health_report() -> Dict[str, Any]:
    checks = {
        "api": _check_api(),
        "redis": _check_redis(),
        "model": _check_model(),
        "llm_weights": _check_llm_weights(),
        "disk": _check_disk(),
    }

    critical_ok = all(
        checks[name]["ok"]
        for name in ("api", "redis", "model", "disk")
    )
    optional_ok = checks["llm_weights"]["ok"]

    if critical_ok and optional_ok:
        status = "ok"
    elif critical_ok:
        status = "degraded"
    else:
        status = "unhealthy"

    return {"status": status, "checks": checks}


def _check_api() -> Dict[str, Any]:
    return {"ok": True, "message": "API process is alive"}


def _check_redis() -> Dict[str, Any]:
    try:
        client = redis.Redis.from_url(CELERY_BROKER_URL, socket_connect_timeout=2, socket_timeout=2)
        client.ping()
        client.close()
        return {"ok": True, "message": "Redis is reachable"}
    except redis.RedisError as exc:
        return {"ok": False, "message": f"Redis unavailable: {exc}"}


def _check_model() -> Dict[str, Any]:
    try:
        from app import pipeline

        model_loaded = pipeline._PIPELINE is not None
    except Exception:
        model_loaded = False

    weights_present = DETECTOR_WEIGHTS_PATH.exists()

    return {
        "ok": weights_present,
        "loaded": model_loaded,
        "weights_present": weights_present,
        "weights_path": str(DETECTOR_WEIGHTS_PATH),
        "message": "Detector weights found" if weights_present else "Detector weights missing",
    }


def _check_llm_weights() -> Dict[str, Any]:
    weights_present = LLM_WEIGHTS_PATH.exists()
    return {
        "ok": weights_present,
        "weights_present": weights_present,
        "weights_path": str(LLM_WEIGHTS_PATH),
        "message": "LLM weights found" if weights_present else "LLM weights missing",
    }


def _check_disk() -> Dict[str, Any]:
    usage = shutil.disk_usage(TMP_DIR)
    ok = usage.free >= MIN_FREE_DISK_BYTES
    return {
        "ok": ok,
        "free_bytes": usage.free,
        "required_free_bytes": MIN_FREE_DISK_BYTES,
        "message": "Enough free disk space" if ok else "Low free disk space",
    }
