from cffi import FFI

ffi = FFI()

ffi.cdef('''
struct RTCube_error
{
	char* message;
};
void RTCube_free_error(struct RTCube_error*);

void RTCube_query(const char* cubesql, struct RTCube_error**);
''')

lib = ffi.dlopen('./lib/librtquery.so')

def query(cubesql):
	errorptr = ffi.new('struct RTCube_error**') # RTCube_error* allocated.
	lib.RTCube_query(cubesql.encode("utf-8"), errorptr)
	error = errorptr[0]
	if error:
		msg = ffi.string(error.message).decode("utf-8")
		lib.RTCube_free_error(error)
		raise ValueError(msg)
