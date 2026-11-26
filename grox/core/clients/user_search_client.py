import os
import time

import grpc
from monitor.metrics import Metrics
from protos.user_search.user_search_pb2 import UserSearchRequest, UserSearchResponse
from protos.user_search.user_search_pb2_grpc import UserSearchServiceStub
from xds_grpc import xds_channel

USER_SEARCH_SERVICE = "user-search.prod.search-ai:grpc"
_LATENCY_BUCKETS_SECONDS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0)


class UserSearchClient:
    _stub: UserSearchServiceStub | None = None

    @classmethod
    async def search(
        cls, request: UserSearchRequest, timeout: float = 5.0
    ) -> UserSearchResponse:
        if cls._stub is None:
            cls._stub = UserSearchServiceStub(xds_channel(USER_SEARCH_SERVICE))
        start = time.perf_counter()
        status = "OK"
        try:
            return await cls._stub.Search(request, timeout=timeout)
        except grpc.aio.AioRpcError as e:
            status = e.code().name
            raise
        finally:
            Metrics.histogram(
                "user_search_client.search.duration",
                explicit_bucket_boundaries_advisory=_LATENCY_BUCKETS_SECONDS,
            ).record(
                time.perf_counter() - start, {"status": status, "pid": str(os.getpid())}
            )
