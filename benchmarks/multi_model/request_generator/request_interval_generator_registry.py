from benchmarks.multi_model.request_generator.gamma_request_interval_generator import (
    GammaRequestIntervalGenerator,
)
from benchmarks.multi_model.request_generator.poisson_request_interval_generator import (
    PoissonRequestIntervalGenerator,
)
from benchmarks.multi_model.request_generator.static_request_interval_generator import (
    StaticRequestIntervalGenerator,
)
from benchmarks.multi_model.request_generator.trace_request_interval_generator import (
    TraceRequestIntervalGenerator,
)
from benchmarks.multi_model.types import RequestIntervalGeneratorType
from benchmarks.multi_model.utils.base_registry import BaseRegistry


class RequestIntervalGeneratorRegistry(BaseRegistry):
    pass


RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.GAMMA, GammaRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.POISSON, PoissonRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.STATIC, StaticRequestIntervalGenerator
)
RequestIntervalGeneratorRegistry.register(
    RequestIntervalGeneratorType.TRACE, TraceRequestIntervalGenerator
)
