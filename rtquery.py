from cffi import FFI
import socket

ffi = FFI()

def deifdef(hpp):
	it = iter(hpp)
	ifdef_level = 0
	while True:
		line = next(it)
		if line.startswith("#ifdef"):
			ifdef_level += 1
		if ifdef_level == 0:
			yield line
		if line.startswith("#endif"):
			ifdef_level -= 1

ffi.cdef("\n".join(deifdef(open("librtquery/query.h"))))

lib = ffi.dlopen('./lib/librtquery.so')

def connect(addresses):
	for addr in addresses:
		yield socket.create_connection(addr)

def query(sockets, cubesql):

	shall_close = False
	if len(sockets):
		try:
			sockets[0].fileno()
		except:
			shall_close = True
			sockets = list(connect(sockets))

	try:
		errorptr = ffi.new('struct RTCube_Error**') # RTCube_Error* allocated.
		lib.RTCube_query([s.fileno() for s in sockets], len(sockets), cubesql.encode("utf-8"), errorptr)
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
