#!/usr/bin/env python3
import sys, subprocess
import signal

class Program(object):
	def main(self, args):
		if len(args) < 2:
			self.usage()
			return 1
		self.args = self.parse_args(args)
		print(self.args)
		self.processes = []
		for id in range(self.args["id"], self.args["id"] + self.args["count"]):
			self.run(id)
		self.wait_all()

	def parse_args(self, args):
		result = { "count": 4, "addresses": "addresses_all_udp", "cubedef":"experiment/cubedef" }
		if len(args) > 1:
			result["id"] = int(args[1])
		if len(args) > 2:
			result["count"] = int(args[2])
		if len(args) > 3:
			result["addresses"] = args[3]
		if len(args) > 4:
			result["cubedef"] = args[4]
		return result

	def run(self, id):
		generator = "bin/data-generator"
		pargs = [str(x) for x in [generator, self.args["addresses"], id, self.args["cubedef"]]]
		print(" ".join(pargs))
		self.processes.append(subprocess.Popen(pargs))
	
	def wait_all(self):
		finished = False
		while not finished:
			try:
				for p in self.processes:
					if p.poll() is None:
						p.wait()
				finished = True
			except KeyboardInterrupt:
				self.kill_all()
	
	def kill_all(self):
		for p in self.processes:
			if not p.poll() is None:
				p.send_signal(signal.SIGINT)
	
	def usage(self):
		print("USAGE: run_generator.py START_ID [COUNT] [ADDRESSES_PATH] [CUBEDEF]")
	
if __name__ == "__main__":
	sys.exit(Program().main(sys.argv))
