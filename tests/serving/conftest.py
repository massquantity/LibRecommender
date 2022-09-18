import pytest
import redis


@pytest.fixture
def redis_client():
    pool = redis.ConnectionPool(host="localhost", port=6379, decode_responses=True)
    r = redis.Redis(connection_pool=pool)
    r.flushdb()
    yield r
    r.flushdb()
    r.close()
