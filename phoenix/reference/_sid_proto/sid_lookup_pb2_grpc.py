# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import grpc
import sid_lookup_pb2 as sid__lookup__pb2

GRPC_GENERATED_VERSION = "1.66.0"
GRPC_VERSION = grpc.__version__
_version_not_supported = False

try:
    from grpc._utilities import first_version_is_lower

    _version_not_supported = first_version_is_lower(GRPC_VERSION, GRPC_GENERATED_VERSION)
except ImportError:
    _version_not_supported = True

if _version_not_supported:
    raise RuntimeError(
        f"The grpc package installed is at version {GRPC_VERSION},"
        + " but the generated code in sid_lookup_pb2_grpc.py depends on"
        + f" grpcio>={GRPC_GENERATED_VERSION}."
        + f" Please upgrade your grpc module to grpcio>={GRPC_GENERATED_VERSION}"
        + f" or downgrade your generated code using grpcio-tools<={GRPC_VERSION}."
    )


class SidLookupServiceStub:
    def __init__(self, channel):
        self.LookupSids = channel.unary_unary(
            "/sid_lookup.SidLookupService/LookupSids",
            request_serializer=sid__lookup__pb2.LookupSidsRequest.SerializeToString,
            response_deserializer=sid__lookup__pb2.LookupSidsResponse.FromString,
            _registered_method=True,
        )


class SidLookupServiceServicer:
    def LookupSids(self, request, context):
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_SidLookupServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "LookupSids": grpc.unary_unary_rpc_method_handler(
            servicer.LookupSids,
            request_deserializer=sid__lookup__pb2.LookupSidsRequest.FromString,
            response_serializer=sid__lookup__pb2.LookupSidsResponse.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "sid_lookup.SidLookupService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
    server.add_registered_method_handlers("sid_lookup.SidLookupService", rpc_method_handlers)


class SidLookupService:
    @staticmethod
    def LookupSids(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/sid_lookup.SidLookupService/LookupSids",
            sid__lookup__pb2.LookupSidsRequest.SerializeToString,
            sid__lookup__pb2.LookupSidsResponse.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
            _registered_method=True,
        )
