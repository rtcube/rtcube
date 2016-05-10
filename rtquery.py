from cffi import FFI
import socket
import sys

ffi = FFI()

def deifdef(hpp):
	it = iter(hpp)
	ifdef_level = 0
	while True:
		line = next(it)
		if line.startswith("#include"):
			continue
		if line.startswith("#ifdef"):
			ifdef_level += 1
		if ifdef_level == 0:
			yield line
		if line.startswith("#endif"):
			ifdef_level -= 1

ffi.cdef("typedef long long int64_t;\n" + "\n".join(deifdef(open("librtquery/query.h"))))

lib = ffi.dlopen('./lib/librtquery.so')

def connect(addresses):
	for addr in addresses:
		try:
			yield socket.create_connection(addr)
		except KeyboardInterrupt:
			raise;
		except:
			print(str(addr[0]) + ":" + str(addr[1]) + " is not responding", file = sys.stderr)

class Cube:
	def __init__(self, c_cube):
		self.c_cube = c_cube

	def __del__(self):
		lib.RTCube_free_cube(self.c_cube)

	def __getitem__(self, index):
		if isinstance(index, tuple):
			index = self.encode_index(index)

		if index >= self.c_cube.data_size:
			raise IndexError

		return self.c_cube.data[index]

	def __len__(self):
		return self.c_cube.data_size

	def __iter__(self):
		for i in range(0, len(self)):
			yield self.decode_index(i), self.c_cube.data[i]

	@property
	def dim_ranges(self):
		return [self.c_cube.dim_ranges[i] for i in range(0, self.c_cube.dim_size)]

	@property
	def mea_types(self):
		return [self.c_cube.mea_types[i] for i in range(0, self.c_cube.mea_size)]

	def encode_index(self, *indexes):
		index = indexes[0]

		for i in range(1, self.c_cube.dim_size):
			index = index * self.c_cube.dim_ranges[i] + indexes[i]

		index = index * self.c_cube.mea_size

		return index

	def decode_index(self, index):
		index = index / self.c_cube.mea_size

		indexes = [0] * self.c_cube.dim_size
	
		for i in range(self.c_cube.dim_size - 1, 0, -1):
			indexes[i] = index % self.c_cube.dim_ranges
			index = index / self.c_cube.dim_ranges

		indexes[0] = index

		return tuple(indexes)

def query(sockets, cubedef, cubesql):

	shall_close = False
	if len(sockets):
		try:
			sockets[0].fileno()
		except:
			shall_close = True
			sockets = list(connect(sockets))

	try:
		errorptr = ffi.new('struct RTCube_Error**') # RTCube_Error* allocated.
		retval = lib.RTCube_query([s.fileno() for s in sockets], len(sockets), cubedef.encode("utf-8"), cubesql.encode("utf-8"), errorptr)
	finally:
		if shall_close:
			for s in sockets:
				s.close()

	error = errorptr[0]
	if error:
		msg = ffi.string(error.message).decode("utf-8")
		type = error.type
		code = error.code
		lib.RTCube_free_error(error)

		if type == lib.InvalidArgument:
			raise ValueError(msg)
		elif type == lib.SystemError:
			raise OSError(code, msg)
		else:
			raise Exception

	return Cube(retval)

"""
import rtquery
x = rtquery.query([("::1", 2001), ("::1", 2002)], "dim time <0,100>; mea content <0,100>.", "SELECT time, COUNT(content)")
for i, v in x:
  print(i, v)
"""
