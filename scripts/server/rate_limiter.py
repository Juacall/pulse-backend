import os
import time
from fastapi import Request, HTTPException, status
from upstash_redis import Redis

# Initialize Upstash REST Client
redis = Redis(
    url=os.getenv("UPSTASH_REDIS_REST_URL"),
    token=os.getenv("UPSTASH_REDIS_REST_TOKEN")
)

# Lua Script for Atomic Token Bucket Evaluation
TOKEN_BUCKET_LUA = """
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local fill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])

local data = redis.call("HMGET", key, "tokens", "last_update")
local tokens = tonumber(data[1])
local last_update = tonumber(data[2])

if not tokens then
    tokens = capacity
    last_update = now
else
    local delta = math.max(0, now - last_update)
    tokens = math.min(capacity, tokens + delta * fill_rate)
end

if tokens >= 1 then
    tokens = tokens - 1
    redis.call("HMSET", key, "tokens", tokens, "last_update", now)
    redis.call("EXPIRE", key, 300) -- TTL of 5 minutes
    return 1
else
    return 0
end
"""

class RateLimiter:
    def __init__(self, requests_per_minute: int, burst_limit: int):
        self.capacity = burst_limit
        self.fill_rate = requests_per_minute / 60.0

    async def __call__(self, request: Request):
        # Identify user via JWT/Session state or fallback to IP address
        user_id = getattr(request.state, "user_id", request.client.host)
        key = f"rate_limit:{request.url.path}:{user_id}"
        now = time.time()

        # Execute Lua script atomically on Upstash
        allowed = redis.eval(
            TOKEN_BUCKET_LUA,
            keys=[key],
            args=[self.capacity, self.fill_rate, now]
        )

        if not allowed:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded. Please wait before making more requests.",
                headers={"Retry-After": "60"}
            )