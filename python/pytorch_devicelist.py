# vim:ts=4 sw=4 et:
import torch

def printgpu( di, cur ):
    print( '%d: %s  %s  %s' % (di, cur, torch.cuda.get_device_name(di), str(torch.cuda.get_device_capability(di))) )

for di in range(torch.cuda.device_count()):
    if torch.cuda.current_device() == di:
        cur= '*'
    else:
        cur=' '
    printgpu( di, cur )
