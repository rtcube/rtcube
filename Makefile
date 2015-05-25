all: compile test

nocuda: compile_nocuda test_nocuda

compile: compile_nocuda lib/librtcudacore.so bin/gpunode

compile_nocuda: bin/send bin/server bin/row_generator lib/librtquery.so lib/librtdummycore.so

test: test_nocuda test_cudacore test_core_cuda

test_nocuda: test_proto test_tokenizer test_parser test_to_ir test_server test_core_dummy

test_proto: bin/tests/test_proto
	./bin/tests/test_proto

test_tokenizer: bin/tests/test_tokenizer
	./bin/tests/test_tokenizer

test_parser: bin/tests/test_parser
	./bin/tests/test_parser

test_to_ir: bin/tests/test_to_ir
	./bin/tests/test_to_ir

test_cudacore: bin/tests/test_cudacore
	./bin/tests/test_cudacore

test_core_dummy: bin/tests/test_core_dummy
	LD_LIBRARY_PATH=./lib ./bin/tests/test_core_dummy

test_core_cuda: bin/tests/test_core_cuda
	LD_LIBRARY_PATH=./lib ./bin/tests/test_core_cuda

test_server: bin/server bin/send
	./bin/server "[::]:2121" &
	./bin/send "[::1]:2121" hello world
	./bin/send "[::1]:2121" DIE

.dirs3:
	mkdir -p bin bin/tests lib obj obj/cudacore
	touch .dirs3

gcc:
	mkdir gcc
	cd gcc; wget https://www.archlinux.org/packages/core/x86_64/gcc/download/ -O gcc.tar.xz
	cd gcc; wget https://www.archlinux.org/packages/core/x86_64/libmpc/download/ -O libmpc.tar.xz
	cd gcc; tar -xvf gcc.tar.xz
	cd gcc; tar -xvf libmpc.tar.xz
	cd gcc; [ -f /usr/lib/x86_64-linux-gnu/crti.o ] && ln -s /usr/lib/x86_64-linux-gnu/crt* usr/lib/ || true

CXX=g++
#Use this to switch to gcc downloaded with make gcc:
#CXX=LD_LIBRARY_PATH=./gcc/usr/lib ./gcc/usr/bin/g++ -static-libgcc
CXX=LD_LIBRARY_PATH=./gcc/usr/lib ./gcc/usr/bin/g++ -static-libgcc

NVCC=nvcc -arch=sm_20 --compiler-options -std=c++11 -U__GXX_EXPERIMENTAL_CXX0X__ -U__cplusplus -D__cplusplus=199711L

CXX14=$(CXX) --std=c++14 -I cxxcompat/include

bin/tests/test_proto: util/* proto/* .dirs3
	$(CXX14) proto/test.cpp proto/proto.cpp -o ./bin/tests/test_proto

bin/tests/test_tokenizer: util/* cubesql/* .dirs3
	$(CXX14) cubesql/test_tokenizer.cpp cubesql/tokenizer.cpp -o ./bin/tests/test_tokenizer

bin/tests/test_parser: util/* cubesql/* .dirs3
	$(CXX14) cubesql/test_parser.cpp cubesql/tokenizer.cpp cubesql/query.cpp cubesql/parser.cpp -o ./bin/tests/test_parser

bin/tests/test_to_ir: util/* cubesql/* proto/* ir/* to_ir/* .dirs3
	$(CXX14) to_ir/test.cpp cubesql/tokenizer.cpp cubesql/query.cpp cubesql/parser.cpp to_ir/cubedef.cpp to_ir/rows.cpp -o ./bin/tests/test_to_ir

bin/send: util/* proto/* send/* .dirs3
	$(CXX14) proto/proto.cpp send/send.cpp -o ./bin/send

bin/server: util/* proto/* server/* .dirs3
	$(CXX14) proto/proto.cpp server/server.cpp -o ./bin/server

bin/row_generator: proto/* row-generator/* .dirs3
	$(CXX14) -lrt proto/proto.cpp row-generator/RowGenerator.cpp -o ./bin/row_generator

lib/librtquery.so: cubesql/* librtquery/* .dirs3
	$(CXX14) -shared -fPIC librtquery/query.cpp cubesql/query.cpp cubesql/tokenizer.cpp cubesql/parser.cpp -o ./lib/librtquery.so.0
	rm -f ./lib/librtquery.so
	ln -s librtquery.so.0 ./lib/librtquery.so

lib/librtdummycore.so: dummycore/* ir/* .dirs3
	$(CXX14) -shared -fPIC dummycore/api.cpp -o ./lib/librtdummycore.so.0
	rm -f ./lib/librtdummycore.so
	ln -s librtdummycore.so.0 ./lib/librtdummycore.so

bin/tests/test_core_dummy: lib/librtdummycore.so ir/* dummycore/api.h test_core.cpp .dirs3
	$(CXX14) -include dummycore/api.h -DCORE_API=DummyCore test_core.cpp -Llib -lrtdummycore -o bin/tests/test_core_dummy

obj/cudacore/RTCube.o: cudacore/*.cuh cudacore/RTCube.cu .dirs3
	$(NVCC) -c --compiler-options -fPIC cudacore/RTCube.cu -o obj/cudacore/RTCube.o

obj/cudacore/RTQuery.o: cudacore/*.cuh cudacore/RTQuery.cu .dirs3
	$(NVCC) -c --compiler-options -fPIC cudacore/RTQuery.cu -o obj/cudacore/RTQuery.o

obj/cudacore/RTUtil.o: cudacore/*.cuh cudacore/RTUtil.cu .dirs3
	$(NVCC) -c --compiler-options -fPIC cudacore/RTUtil.cu -o obj/cudacore/RTUtil.o

obj/cudacore/RTCubeApi.o: ir/*.h cudacore/*.h cudacore/*.cuh cudacore/RTCubeApi.cu .dirs3
	$(NVCC) -c --compiler-options -fPIC cudacore/RTCubeApi.cu -o obj/cudacore/RTCubeApi.o

obj/cudacore/api.o: ir/*.h cudacore/*.h cudacore/*.cuh cudacore/api.cu .dirs3
	$(NVCC) -c --compiler-options -fPIC cudacore/api.cu -o obj/cudacore/api.o

obj/cudacore/sample.o: cudacore/*.cuh cudacore/sample.cu .dirs3
	$(NVCC) -c cudacore/sample.cu -o obj/cudacore/sample.o

bin/tests/test_cudacore: obj/cudacore/RTCube.o obj/cudacore/RTQuery.o obj/cudacore/RTUtil.o obj/cudacore/sample.o
	$(NVCC) obj/cudacore/RTCube.o obj/cudacore/RTQuery.o obj/cudacore/RTUtil.o obj/cudacore/sample.o -o bin/tests/test_cudacore

lib/librtcudacore.so: obj/cudacore/RTCube.o obj/cudacore/RTQuery.o obj/cudacore/RTCubeApi.o obj/cudacore/RTUtil.o obj/cudacore/api.o .dirs3
	$(NVCC) -shared --compiler-options -fPIC obj/cudacore/RTCube.o obj/cudacore/RTQuery.o obj/cudacore/RTCubeApi.o obj/cudacore/RTUtil.o obj/cudacore/api.o -o ./lib/librtcudacore.so.0
	rm -f ./lib/librtcudacore.so
	ln -s librtcudacore.so.0 ./lib/librtcudacore.so

bin/tests/test_core_cuda: lib/librtcudacore.so ir/* cudacore/api.h test_core.cpp .dirs3
	$(CXX14) -include cudacore/api.h -DCORE_API=CudaCore test_core.cpp -Llib -lrtcudacore -o bin/tests/test_core_cuda

bin/gpunode: lib/librtcudacore.so gpunode/* util/* proto/* server/* .dirs3
	$(CXX14) proto/proto.cpp gpunode/main.cpp gpunode/RTServer.cpp -Llib -lrtcudacore -o bin/gpunode
