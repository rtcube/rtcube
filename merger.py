#!/usr/bin/env python3
import sys
import glob

class Program(object):
	def main(self, argv):
		self.name = argv[1]
		self.qfile = "querer_" + self.name + ".txt"
		self.gfile = "generator_" + self.name + "_9.txt"
		self.ofile = "results_" + self.name + ".txt"
		dvals = {}
		for qline in open(self.qfile, "r"):
			linevals = qline.strip().split(",")
			key = int(linevals[0])
			value = int(linevals[1])
			if not key in dvals:
				dvals[key] = [key, value]
		for gline in open(self.gfile, "r"):
			linevals = gline.strip().split(",")
			key = int(linevals[1])
			value = int(linevals[3])
			if key in dvals:
				dvals[key].append(value)
		dkeys = [x for x in dvals.keys()]
		dkeys.sort()
		with open(self.ofile, "w") as file:
			for key in dkeys:
				print(",".join(str(x) for x in dvals[key]), file = file)
		return 0
		

if __name__ == "__main__":
	sys.exit(Program().main(sys.argv))
