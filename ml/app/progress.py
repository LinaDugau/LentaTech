from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Dict

import redis
import redis.asyncio as aioredis

from app.config import REDIS_PROGRESS_URL

logger = logging.getLogger(__name__)


def _channel(job_id: str) -> str:
    return f"job_progress:{job_id}"


def publish_job_progress(job_id: str, payload: Dict[str, Any]) -> None:
    try:
        client = redis.Redis.from_url(REDIS_PROGRESS_URL, decode_responses=True)
        client.publish(_channel(job_id), json.dumps(payload, ensure_ascii=False))
        client.close()
    except redis.RedisError as exc:
        logger.warning("Failed to publish progress for job %s: %s", job_id, exc)


async def get_progress_stream(job_id: str) -> AsyncIterator[Dict[str, Any]]:
    client = aioredis.Redis.from_url(REDIS_PROGRESS_URL, decode_responses=True)
    pubsub = client.pubsub()

    try:
        await pubsub.subscribe(_channel(job_id))
        async for message in pubsub.listen():
            if message.get("type") != "message":
                continue

            try:
                payload = json.loads(message["data"])
            except (TypeError, json.JSONDecodeError):
                logger.warning("Invalid progress payload for job %s: %r", job_id, message["data"])
                continue

            yield payload

            if payload.get("status") in {"done", "error"}:
                break
    finally:
        await pubsub.unsubscribe(_channel(job_id))
        await pubsub.close()
        await client.close()
