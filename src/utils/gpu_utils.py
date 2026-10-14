import os
import GPUtil
from time import sleep

def get_available_gpu(memory_limit=0.91):
	if os.environ.get("FEDLN_DEVICE") == "cpu":
		return "-1"
	# Nousphera maintenance: absent NVIDIA hardware cannot become free by waiting.
	if not GPUtil.getGPUs():
		raise RuntimeError("No NVIDIA GPU detected by GPUtil. These experiment scripts require a working NVIDIA driver; CPU/Apple GPU training has not been validated.")
	# Wait for GPU memory release.
	while len(GPUtil.getAvailable(order='memory', limit=len(GPUtil.getGPUs()), maxLoad=1.0, maxMemory=memory_limit)) == 0: sleep(1)
	cuda_device_ids = GPUtil.getAvailable(order='memory', limit=len(GPUtil.getGPUs()), maxLoad=1.0, maxMemory=memory_limit)
	cuda_device_ids.extend("") # Fix no gpu issue
	return str(cuda_device_ids[0])
