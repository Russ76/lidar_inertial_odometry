#!/usr/bin/env python3
import time

def get_cpu_usage():
    with open('/proc/stat', 'r') as f:
        line = f.readline()
    parts = line.split()
    user, nice, system, idle = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
    return user, nice, system, idle

prev = get_cpu_usage()
time.sleep(0.1)

while True:
    curr = get_cpu_usage()
    
    user_diff = curr[0] - prev[0]
    nice_diff = curr[1] - prev[1]
    system_diff = curr[2] - prev[2]
    idle_diff = curr[3] - prev[3]
    
    total = user_diff + nice_diff + system_diff + idle_diff
    if total > 0:
        usage = (user_diff + nice_diff + system_diff) * 100.0 / total
    else:
        usage = 0
    
    print(f"CPU: {usage:.1f}%")
    
    prev = curr
    time.sleep(1)
