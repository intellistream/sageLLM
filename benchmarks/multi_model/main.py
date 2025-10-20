import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from benchmarks.multi_model.utils.random import set_seeds
from benchmarks.multi_model.request_generator import RequestGeneratorRegistry
from benchmarks.multi_model.config import (
    SyntheticRequestGeneratorConfig
)

models: list[str] =[
    "Qwen3-4B",
    "llama",
]

def main():
    set_seeds(25)
    generator_config = SyntheticRequestGeneratorConfig()
    generator_config.seed = 90
    generator_config.models = models
    request_generator = RequestGeneratorRegistry.get(
        generator_config.get_type(),
        generator_config,
    )
    requests =request_generator.generate()
    for request in requests:
        print(request)

if __name__ == '__main__':
    main()
