import sys

sys.path.append("../")
import rtquery as rtq
import time


addresses = [
    ("192.168.143.217", 50000),
    ("192.168.143.217", 60000)
]

# TODO
"""
Here comes list of addresses (most likely plain IPv4) of gpunodes in 304
"""

cubedef = open("experiment/cubedef").read()

query = "SELECT MAX(m1), MAX(m2)"

current_time_ms = lambda: int(round(time.time() * 1000))

while True:
    start_time = current_time_ms()
    res = rtq.query(addresses, cubedef, query)
    query_time = current_time_ms() - start_time

    # TODO - ms?
    print("result:", res[0], "time[ms]:", query_time)
