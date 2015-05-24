from cffi import FFI

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

def query(cubesql):
	errorptr = ffi.new('struct RTCube_error**') # RTCube_error* allocated.
	lib.RTCube_query([], 0, cubesql.encode("utf-8"), errorptr)
	error = errorptr[0]
	if error:
		msg = ffi.string(error.message).decode("utf-8")
		lib.RTCube_free_error(error)
		raise ValueError(msg)
