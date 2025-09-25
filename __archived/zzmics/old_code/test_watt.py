from pynvml_utils import nvidia_smi
import time
import threading
from halib import *

nvsmi = nvidia_smi.getInstance()
result = nvsmi.DeviceQuery("memory.free, memory.total")
gpu_stats = []
running_monitor = True


def monitor(interval=0.01):  # 10ms sampling
    """
    {
    │   'fb_memory_usage': {'used': 2757.10546875, 'unit': 'MiB'},
    │   'power_readings': {'power_draw': 126.495, 'power_state': 'P0', 'unit': 'W'}
    }
    """
    while running_monitor:
        stats = nvsmi.DeviceQuery("power.draw, memory.used")["gpu"][0]
        pprint(stats)
        gpu_stats.append(
            {
                "power": stats["power_readings"]["power_draw"],
                "power_unit": stats["power_readings"]["unit"],
                "memory": stats["fb_memory_usage"]["used"],
                "memory_unit": stats["fb_memory_usage"]["unit"],
            }
        )
        time.sleep(interval)


# Start monitoring thread
monitor_thread = threading.Thread(target=monitor)
monitor_thread.start()

# --- Run your DL inference here ---
start = time.perf_counter()

time.sleep(0.3)  # Replace with model inference

end = time.perf_counter()

# Stop monitoring
running_monitor = False
monitor_thread.join()

# Analyze results
powers = [s["power"] for s in gpu_stats if s["power"] is not None]
memories = [s["memory"] for s in gpu_stats if s["memory"] is not None]

avg_power = sum(powers) / len(powers) if powers else 0
max_memory = max(memories) if memories else 0
power_unit = gpu_stats[0]["power_unit"] if gpu_stats else "W"
memory_unit = gpu_stats[0]["memory_unit"] if gpu_stats else "MiB"


print(f"Average GPU power draw: {avg_power:.2f} {power_unit}")
print(f"Peak GPU memory usage: {max_memory} {memory_unit}")
