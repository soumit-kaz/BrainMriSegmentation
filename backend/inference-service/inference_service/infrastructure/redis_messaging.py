import json
import time
from collections.abc import AsyncGenerator
from dataclasses import asdict
from typing import Any, cast

from redis import Redis
from redis.asyncio import Redis as AsyncRedis
from redis.exceptions import RedisError, ResponseError

from inference_service.configuration import InferenceServiceSettings
from inference_service.domain.segmentation_job import ModelInformation
from inference_service.helpers import get_job_event_channel

JOB_ID_FIELD = "job_id"
EXISTING_GROUP_ERROR_PREFIX = "BUSYGROUP"
MODEL_INFORMATION_KEY = "inference-worker:model-information"
PROCESSING_DURATION_KEY = "inference-worker:average-processing-seconds"
RATE_LIMIT_KEY_PREFIX = "rate-limit"
RATE_LIMIT_WINDOW_SECONDS = 60
PROCESSING_DURATION_SMOOTHING = 0.2


def create_redis_client(redis_url: str) -> Redis:
    return Redis.from_url(redis_url, decode_responses=True)


def create_async_redis_client(redis_url: str) -> AsyncRedis:
    return AsyncRedis.from_url(redis_url, decode_responses=True)


class RedisJobQueue:
    def __init__(self, redis_client: Redis, settings: InferenceServiceSettings):
        self.redis_client = redis_client
        self.stream_name = settings.job_stream_name
        self.consumer_group = settings.job_consumer_group
        self.consumer_name = settings.worker_name
        self.maximum_stream_length = settings.maximum_stream_length

    def is_available(self) -> bool:
        try:
            return bool(self.redis_client.ping())
        except RedisError:
            return False

    def ensure_consumer_group(self) -> None:
        try:
            self.redis_client.xgroup_create(self.stream_name, self.consumer_group, id="0", mkstream=True)
        except ResponseError as error:
            if not str(error).startswith(EXISTING_GROUP_ERROR_PREFIX):
                raise

    def add_job(self, job_id: str) -> None:
        self.redis_client.xadd(self.stream_name, {JOB_ID_FIELD: job_id}, maxlen=self.maximum_stream_length, approximate=True)

    def get_next_job(self, block_milliseconds: int) -> tuple[str, str] | None:
        stream_entries = cast(list[Any], self.redis_client.xreadgroup(
            self.consumer_group, self.consumer_name, {self.stream_name: ">"}, count=1, block=block_milliseconds
        ))
        if not stream_entries:
            return None
        _, messages = stream_entries[0]
        message_id, fields = messages[0]
        return message_id, fields[JOB_ID_FIELD]

    def get_stalled_job(self, minimum_idle_milliseconds: int) -> tuple[str, str] | None:
        claim_result = cast(list[Any], self.redis_client.xautoclaim(
            self.stream_name, self.consumer_group, self.consumer_name, min_idle_time=minimum_idle_milliseconds, start_id="0-0", count=1
        ))
        claimed_messages = claim_result[1] if len(claim_result) > 1 else []
        for message_id, fields in claimed_messages:
            if fields and JOB_ID_FIELD in fields:
                return message_id, fields[JOB_ID_FIELD]
        return None

    def acknowledge(self, message_id: str) -> None:
        self.redis_client.xack(self.stream_name, self.consumer_group, message_id)


class RedisJobEventPublisher:
    def __init__(self, redis_client: Redis, channel_prefix: str):
        self.redis_client = redis_client
        self.channel_prefix = channel_prefix

    def publish(self, job_id: str, event: dict[str, Any]) -> None:
        self.redis_client.publish(get_job_event_channel(self.channel_prefix, job_id), json.dumps({**event, JOB_ID_FIELD: job_id}))


class RedisJobEventSubscriber:
    def __init__(self, async_redis_client: AsyncRedis, channel_prefix: str):
        self.async_redis_client = async_redis_client
        self.channel_prefix = channel_prefix

    async def get_events(self, job_id: str) -> AsyncGenerator[dict[str, Any], None]:
        publish_subscribe = self.async_redis_client.pubsub()
        await publish_subscribe.subscribe(get_job_event_channel(self.channel_prefix, job_id))
        try:
            async for message in publish_subscribe.listen():
                if message.get("type") == "message":
                    yield json.loads(message["data"])
        finally:
            await publish_subscribe.unsubscribe()
            await publish_subscribe.aclose()


class RedisRateLimiter:
    def __init__(self, redis_client: Redis):
        self.redis_client = redis_client

    def is_allowed(self, client_key: str, limit_per_minute: int) -> bool:
        window_index = int(time.time() // RATE_LIMIT_WINDOW_SECONDS)
        window_key = f"{RATE_LIMIT_KEY_PREFIX}:{client_key}:{window_index}"
        pipeline = self.redis_client.pipeline()
        pipeline.incr(window_key)
        pipeline.expire(window_key, RATE_LIMIT_WINDOW_SECONDS)
        request_count = int(pipeline.execute()[0])
        return request_count <= limit_per_minute


class RedisWorkerStatusStore:
    def __init__(self, redis_client: Redis, status_ttl_seconds: int):
        self.redis_client = redis_client
        self.status_ttl_seconds = status_ttl_seconds

    def set_model_information(self, model_information: ModelInformation) -> None:
        self.redis_client.set(MODEL_INFORMATION_KEY, json.dumps(asdict(model_information)), ex=self.status_ttl_seconds)

    def get_model_information(self) -> ModelInformation | None:
        serialized_information = cast(str | None, self.redis_client.get(MODEL_INFORMATION_KEY))
        return ModelInformation(**json.loads(serialized_information)) if serialized_information else None

    def add_processing_duration(self, duration_seconds: float) -> None:
        previous_average = cast(str | None, self.redis_client.get(PROCESSING_DURATION_KEY))
        updated_average = duration_seconds if previous_average is None else \
            (1 - PROCESSING_DURATION_SMOOTHING) * float(previous_average) + PROCESSING_DURATION_SMOOTHING * duration_seconds
        self.redis_client.set(PROCESSING_DURATION_KEY, updated_average)

    def get_average_processing_seconds(self, default_seconds: float) -> float:
        stored_average = cast(str | None, self.redis_client.get(PROCESSING_DURATION_KEY))
        return float(stored_average) if stored_average is not None else default_seconds
