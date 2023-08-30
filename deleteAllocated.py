# import gc
# del variables
# gc.collect()

import torch
# torch.cuda.empty_cache()
import sys
import gc
import os
# import psutil
# gc.collect()
# torch.cuda.empty_cache()

# def memReport():
#     for obj in gc.get_objects():
#         if torch.is_tensor(obj):
#             print(type(obj), obj.size())
    
# def cpuStats():
#         print(sys.version)
#         print(psutil.cpu_percent())
#         print(psutil.virtual_memory())  # physical memory usage
#         pid = os.getpid()
#         py = psutil.Process(pid)
#         memoryUse = py.memory_info()[0] / 2. ** 30  # memory use in GB...I think
#         print('memory GB:', memoryUse)

# cpuStats()
# memReport()

for epoch in range(20):
    torch.cuda.empty_cache()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

