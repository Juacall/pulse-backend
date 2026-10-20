import logging
import json
import os
from typing import Any, Optional
from upstash_redis import Redis

from scripts.rag.environment import load_project_environment

load_project_environment()

logger = logging.getLogger("uvicorn.error")

class SafeRedisCache:
    def __init__(self):
        url = os.getenv("UPSTASH_REDIS_REST_URL")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

        if not url or not token:
            logger.warning("Upstash Redis environment variables missing. Cache disabled.")
            self.client = None
        else:
            try:
                self.client = Redis(url=url, token=token)
            except Exception as e:
                logger.error(f"Failed to initialize Upstash Redis client: {e}")
                self.client = None

    def get(self, key: str) -> Optional[Any]:
        """Safely fetch a key from Redis. Returns None on cache miss or connection error."""
        if not self.client:
            return None
        try:
            val = self.client.get(key)
            if val is None:
                return None
            return json.loads(val) if isinstance(val, str) else val
        except Exception as e:
            logger.warning(f"Redis READ error for key '{key}': {e}. Falling back to primary DB.")
            return None

    def set(self, key: str, value: Any, ex_seconds: int = 86400) -> bool:
        """Safely set a key in Redis with a TTL. Returns True if written, False on failure."""
        if not self.client:
            return False
        try:
            serialized = json.dumps(value)
            self.client.set(key, serialized, ex=ex_seconds)
            return True
        except Exception as e:
            logger.warning(f"Redis WRITE error for key '{key}': {e}. Continuing without caching.")
            return False

# Export a single global instance
cache = SafeRedisCache()