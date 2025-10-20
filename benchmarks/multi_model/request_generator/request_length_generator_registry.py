from benchmarks.multi_model.request_generator.fixed_request_length_generator import (
    FixedRequestLengthGenerator,
)
from benchmarks.multi_model.request_generator.trace_request_length_generator import (
    TraceRequestLengthGenerator,
)
from benchmarks.multi_model.request_generator.uniform_request_length_generator import (
    UniformRequestLengthGenerator,
)
from benchmarks.multi_model.request_generator.zipf_request_length_generator import (
    ZipfRequestLengthGenerator,
)
from benchmarks.multi_model.types import RequestLengthGeneratorType
from benchmarks.multi_model.utils.base_registry import BaseRegistry


class RequestLengthGeneratorRegistry(BaseRegistry):
    pass


RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.ZIPF, ZipfRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.UNIFORM, UniformRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.TRACE, TraceRequestLengthGenerator
)
RequestLengthGeneratorRegistry.register(
    RequestLengthGeneratorType.FIXED, FixedRequestLengthGenerator
)
