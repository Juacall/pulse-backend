import logging
import json
import os
from typing import Any, Optional, Union, List
import httpx
from upstash_redis import Redis

from scripts.rag.environment import load_project_environment

load_project_environment()

logger = logging.getLogger("uvicorn.error")
REDIS_TIMEOUT_SECONDS = 2.0
# Unset ENV falls back to "prod"; set ENV explicitly in every non-prod environment
ENV = (os.getenv("ENV") or "prod").strip().lower()


def env_key(key: str) -> str:
    """Namespaces a cache key by environment so non-prod writes never collide with prod."""
    prefix = f"{ENV}:"
    return key if key.startswith(prefix) else f"{prefix}{key}"


class SafeRedisCache:
    def __init__(self):
        url = os.getenv("UPSTASH_REDIS_REST_URL")
        token = os.getenv("UPSTASH_REDIS_REST_TOKEN")

        if not url or not token:
            logger.warning("Upstash Redis environment variables missing. Cache disabled.")
            self.client = None
        else:
            try:
                self.client = Redis(url=url, token=token, rest_retries=0)
                # The Upstash SDK defaults to no HTTP timeout; bound it so a slow Redis can't hang requests
                self.client._http._client.timeout = httpx.Timeout(REDIS_TIMEOUT_SECONDS)
            except Exception as e:
                logger.error(f"Failed to initialize Upstash Redis client: {e}")
                self.client = None

    def get(self, key: str) -> Optional[Any]:
        """Safely fetch a key from Redis. Returns None on cache miss or connection error."""
        if not self.client:
            return None
        try:
            val = self.client.get(env_key(key))
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
            serialized = json.dumps(value, default=str)
            self.client.set(env_key(key), serialized, ex=ex_seconds)
            return True
        except Exception as e:
            logger.warning(f"Redis WRITE error for key '{key}': {e}. Continuing without caching.")
            return False

    def delete(self, *keys: Union[str, List[str]]) -> int:
        """
        Safely invalidates one or more keys in Upstash Redis.
        Accepts individual key strings or a list of key strings.
        Returns the number of keys deleted (or 0 on failure).
        """
        if not self.client or not keys:
            return 0

        try:
            # Flatten keys if passed as a single list or tuple
            flat_keys = []
            for k in keys:
                if isinstance(k, (list, tuple)):
                    flat_keys.extend(k)
                elif k:
                    flat_keys.append(str(k))

            if not flat_keys:
                return 0
            flat_keys = [env_key(k) for k in flat_keys]

            # Upstash Redis SDK accepts *flat_keys positional arguments
            deleted_count = self.client.delete(*flat_keys)
            logger.info(f"Successfully invalidated Redis keys: {flat_keys}")
            return deleted_count or 0
        except Exception as err:
            logger.warning(f"Redis DELETE error for keys={keys}: {err}. Continuing without cache eviction.")
            return 0

    def invalidate_user_cache(self, user_id: str):
        """Purges cached user summary keys from Redis."""
        try:
            self.delete(f"user:{user_id}:summary")
            logger.info(f"Successfully invalidated Redis cache for user_id={user_id}")
        except Exception as err:
            logger.warning(f"Failed to invalidate Redis cache for user_id={user_id}: {err}")

# Export a single global instance
cache = SafeRedisCache()