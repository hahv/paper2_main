# install `pynvml_utils` package first
# see this repo: https://github.com/gpuopenanalytics/pynvml
from pynvml_utils import nvidia_smi
import time
import threading
from rich.pretty import pprint

class GPUMonitor:
    def __init__(self, gpu_index=0, interval=0.01):
        self.nvsmi = nvidia_smi.getInstance()
        self.gpu_index = gpu_index
        self.interval = interval
        self.gpu_stats = []
        self._running = False
        self._thread = None

    def _monitor(self):
        while self._running:
            stats = self.nvsmi.DeviceQuery("power.draw, memory.used")["gpu"][
                self.gpu_index
            ]
            # pprint(stats)
            self.gpu_stats.append(
                {
                    "power": stats["power_readings"]["power_draw"],
                    "power_unit": stats["power_readings"]["unit"],
                    "memory": stats["fb_memory_usage"]["used"],
                    "memory_unit": stats["fb_memory_usage"]["unit"],
                }
            )
            time.sleep(self.interval)

    def start(self):
        if not self._running:
            self._running = True
            # clear previous stats
            self.gpu_stats.clear()
            self._thread = threading.Thread(target=self._monitor)
            self._thread.start()

    def stop(self):
        if self._running:
            self._running = False
            self._thread.join()
            # clear the thread reference
            self._thread = None

    def get_stats(self):
        ## return self.gpu_stats
        assert self._running is False, "GPU monitor is still running. Stop it first."

        powers = [s["power"] for s in self.gpu_stats if s["power"] is not None]
        memories = [s["memory"] for s in self.gpu_stats if s["memory"] is not None]
        avg_power = sum(powers) / len(powers) if powers else 0
        max_memory = max(memories) if memories else 0
        # power_unit = self.gpu_stats[0]["power_unit"] if self.gpu_stats else "W"
        # memory_unit = self.gpu_stats[0]["memory_unit"] if self.gpu_stats else "MiB"
        return {"gpu_avg_power": avg_power, "gpu_avg_max_memory": max_memory}
