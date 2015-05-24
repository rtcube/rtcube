all: compile test

compile: bin/send bin/server bin/row_generator lib/librtquery.so bin/gpunode

test: test_proto test_tokenizer test_parser test_to_ir test_server test_cudacore

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

test_server: bin/server bin/send
	./bin/server "[::]:2121" &
	./bin/send "[::1]:2121" hello world
	./bin/send "[::1]:2121" DIE

.dirs3:
	mkdir -p bin bin/tests lib obj
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

NVCC=nvcc -arch=sm_20

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

obj/RTCube.o: gpunode/*.h gpunode/*.cuh gpunode/RTCube.cu ir/* .dirs3
	$(NVCC) -c gpunode/RTCube.cu -o obj/RTCube.o

obj/RTQuery.o: gpunode/*.h gpunode/*.cuh gpunode/RTQuery.cu ir/* .dirs3
	$(NVCC) -c gpunode/RTQuery.cu -o obj/RTQuery.o

obj/RTUtil.o: gpunode/*.h gpunode/*.cuh gpunode/RTUtil.cu ir/* .dirs3
	$(NVCC) -c gpunode/RTUtil.cu -o obj/RTUtil.o

obj/RTCubeApi.o: gpunode/*.h  gpunode/*.cuh gpunode/RTCubeApi.cu ir/* .dirs3
	$(NVCC) -c gpunode/RTCubeApi.cu -o obj/RTCubeApi.o

obj/sample.o: gpunode/*.h  gpunode/*.cuh gpunode/sample.cu ir/* .dirs3
	$(NVCC) -c gpunode/sample.cu -o obj/sample.o

bin/tests/test_cudacore: obj/RTCube.o obj/RTQuery.o obj/RTUtil.o obj/sample.o
	$(NVCC) obj/RTCube.o obj/RTQuery.o obj/RTUtil.o obj/sample.o -o bin/tests/test_cudacore

bin/gpunode: obj/RT*.o util/* proto/* server/* .dirs3
	$(CXX14) proto/proto.cpp gpunode/main.cpp gpunode/RTServer.cpp obj/RTCube.o obj/RTQuery.o obj/RTUtil.o obj/RTCubeApi.o /usr/local/cuda/lib64/libcudart_static.a -pthread -ldl -lrt -o bin/gpunode
