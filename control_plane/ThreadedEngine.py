import asyncio
import threading
from typing import Dict, List, Any

from vllm import LLMEngine, EngineArgs, SamplingParams, RequestOutput


class ThreadedLLMEngine:

    def __init__(self, engine_args: EngineArgs):
        self.engine = LLMEngine.from_engine_args(engine_args)
        self.loop = asyncio.get_running_loop()
        self.request_futures: Dict[str, asyncio.Future] = {}
        self.driver_thread = threading.Thread(target=self._driver_loop, name=f"Driver-{engine_args.model}")
        self.driver_thread.daemon = True
        self._running = True
        self.driver_thread.start()

    def _driver_loop(self):
        while self._running:
            request_outputs = self.engine.step()
            for output in request_outputs:
                future = self.request_futures.get(output.request_id)
                if future and not future.done() and output.finished:
                    final_text = output.outputs[0].text
                    self.loop.call_soon_threadsafe(future.set_result, final_text)

    def stop(self):
        self._running = False
        self.driver_thread.join()

    async def submit_request(self, request_id: str, prompt: str, sampling_params: SamplingParams) -> str:
        future = self.loop.create_future()
        self.request_futures[request_id] = future
        self.engine.add_request(request_id, prompt, sampling_params)
        try:
            return await asyncio.wait_for(future, timeout=120.0)
        finally:
            self.request_futures.pop(request_id, None)