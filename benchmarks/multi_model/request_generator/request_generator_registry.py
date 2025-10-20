from benchmarks.multi_model.request_generator.synthetic_request_generator import (
    SyntheticRequestGenerator,
)
from benchmarks.multi_model.request_generator.trace_request_generator import (
    TraceRequestGenerator,
)
from benchmarks.multi_model.types import RequestGeneratorType
from benchmarks.multi_model.utils.base_registry import BaseRegistry


class RequestGeneratorRegistry(BaseRegistry):
    pass


RequestGeneratorRegistry.register(
    RequestGeneratorType.SYNTHETIC, SyntheticRequestGenerator
)
RequestGeneratorRegistry.register(RequestGeneratorType.TRACE, TraceRequestGenerator)
