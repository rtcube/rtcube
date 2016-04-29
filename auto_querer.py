#!/usr/bin/env python3
import sys

sys.path.append("../")
import rtquery as rtq
import time

def print_usage():
	print("USAGE: auto_querer.py <addresses_file> <cubedef_file>")

def parse_addresses(file):
	with open(file) as f:
		 stripped = [x.strip() for x in f.readlines()]
		 split = [x.split(':') for x in stripped]

		 return [(x[0], x[1]) for x in split]

def main():
	if len(sys.argv) < 3:
		print_usage()
		sys.exit()

	addresses_file = sys.argv[1]
	cubedef = open(sys.argv[2]).read()
	current_time_ms = lambda: int(round(time.time() * 1000))
	query = "SELECT MAX(m1), MAX(m2)"

	addresses = parse_addresses(addresses_file)

	stderrleft = 5
	while True:
		#start_time = current_time_ms()
		res = rtq.query(addresses, cubedef, query)
		#query_time = current_time_ms() - start_time

		# TODO - ms?
		line = str(res[0]) + "," + str(current_time_ms())
		print(line)
		if stderrleft > 0:
			print(line, file = sys.stderr)
			stderrleft-=1
		# print("result:", res[0], "time[ms]:", query_time)

if __name__ == "__main__":
	try:
		main()
	except KeyboardInterrupt:
		print("Terminated by user", file = sys.stderr)
		sys.exit(1)
