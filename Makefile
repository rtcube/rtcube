all: test_proto test_tokenizer test_parser test_to_ir test_cudacore bin/send bin/server bin/row_generator

test: test_proto test_tokenizer test_parser test_to_ir test_cudacore test_server

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

.dirs2:
	mkdir -p bin bin/tests obj
	touch .dirs2

gcc:
	mkdir gcc
	cd gcc; wget https://www.archlinux.org/packages/core/x86_64/gcc/download/ -O gcc.tar.xz
	cd gcc; wget https://www.archlinux.org/packages/core/x86_64/libmpc/download/ -O libmpc.tar.xz
	cd gcc; tar -xvf gcc.tar.xz
	cd gcc; tar -xvf libmpc.tar.xz
	cd gcc; [ -f /usr/lib/x86_64-linux-gnu/crti.o ] && ln -s /usr/lib/x86_64-linux-gnu/crt* usr/lib/ || true

CXX=g++
#Use this to switch to gcc downloaded with make gcc:
CXX=LD_LIBRARY_PATH=./gcc/usr/lib ./gcc/usr/bin/g++ -static-libgcc

NVCC=nvcc -arch=sm_20

CXX14=$(CXX) --std=c++14 -I cxxcompat/include

bin/tests/test_proto: util/* proto/* .dirs2
	$(CXX14) proto/test.cpp proto/proto.cpp -o ./bin/tests/test_proto

bin/tests/test_tokenizer: util/* cubesql/* .dirs2
	$(CXX14) cubesql/test_tokenizer.cpp cubesql/tokenizer.cpp -o ./bin/tests/test_tokenizer

bin/tests/test_parser: util/* cubesql/* .dirs2
	$(CXX14) cubesql/test_parser.cpp cubesql/tokenizer.cpp cubesql/query.cpp cubesql/parser.cpp -o ./bin/tests/test_parser

bin/tests/test_to_ir: util/* cubesql/* proto/* ir/* to_ir/* .dirs2
	$(CXX14) to_ir/test.cpp cubesql/tokenizer.cpp cubesql/query.cpp cubesql/parser.cpp to_ir/cubedef.cpp to_ir/rows.cpp -o ./bin/tests/test_to_ir

bin/send: util/* proto/* send/* .dirs2
	$(CXX14) proto/proto.cpp send/send.cpp -o ./bin/send

bin/server: util/* proto/* server/* .dirs2
	$(CXX14) proto/proto.cpp server/server.cpp -o ./bin/server

bin/row_generator: proto/* row-generator/* .dirs2
	$(CXX14) proto/proto.cpp row-generator/RowGenerator.cpp -o ./bin/row_generator

obj/RTCube.o: gpunode-cudacore/* ir/* .dirs2
	$(NVCC) -c gpunode-cudacore/RTCube.cu -o obj/RTCube.o

obj/RTQuery.o: gpunode-cudacore/* ir/* .dirs2
	$(NVCC) -c gpunode-cudacore/RTQuery.cu -o obj/RTQuery.o

bin/tests/test_cudacore: obj/RTCube.o obj/RTQuery.o
	$(NVCC) obj/RTCube.o obj/RTQuery.o gpunode-cudacore/main.cu -o bin/tests/test_cudacore

frankenfile: obj/RTCube.o obj/RTQuery.o util/* proto/* server/* .dirs2
	$(CXX14) proto/proto.cpp server/server.cpp obj/RTCube.o obj/RTQuery.o /usr/local/cuda/lib64/libcudart_static.a -pthread -ldl -lrt -o frankenfile
