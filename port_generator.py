#!/usr/bin/env python3

class Program(object):	
	def main(self):
		self.infilename = "addresses_all"
		self.tcpfilename = self.infilename + "_tcp"
		self.udpfilename = self.infilename + "_udp"
		with open(self.tcpfilename, "w") as tcpfile:
			with open(self.udpfilename, "w") as udpfile:
				for line in open(self.infilename, "r"):
					print(line.strip('\n') + ":50000", file = tcpfile)
					print(line.strip('\n') + ":50001", file = udpfile)

if __name__ == "__main__":
	Program().main()
