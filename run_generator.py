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
		for id in range(self.args["id"], self.args["id"] + self.args["count"] * self.args["step"], self.args["step"]):
			self.run(id)
		self.wait_all()

	def parse_args(self, args):
		result = { "count": 4, "addresses": "addresses_all_udp", "cubedef":"experiment/cubedef", "step": 1 }
		if len(args) > 1:
			result["id"] = int(args[1])
		if len(args) > 2:
			result["step"] = int(args[2])
		if len(args) > 3:
			result["count"] = int(args[3])
		if len(args) > 4:
			result["addresses"] = args[4]
		if len(args) > 5:
			result["cubedef"] = args[5]
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
		print("USAGE: run_generator.py START_ID [ID_STEP = 1] [COUNT = 4] [ADDRESSES_PATH = addresses_all_udp] [CUBEDEF = experiment/cubedef]")
	
if __name__ == "__main__":
	sys.exit(Program().main(sys.argv))
