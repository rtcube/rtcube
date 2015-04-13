all: test_proto test_tokenizer bin/send bin/server

test: test_proto test_tokenizer test_server

test_proto: bin/tests/test_proto
	./bin/tests/test_proto

test_tokenizer: bin/tests/test_tokenizer
	./bin/tests/test_tokenizer

test_server: bin/server bin/send
	./bin/server "[::]:2121" &
	./bin/send "[::1]:2121" hello world
	./bin/send "[::1]:2121" DIE

.dirs:
	mkdir -p bin bin/tests
	touch .dirs

bin/tests/test_proto: util/* proto/* .dirs
	g++ --std=c++14 -I cxxcompat/include proto/test.cpp proto/proto.cpp -o ./bin/tests/test_proto

bin/tests/test_tokenizer: util/* cubesql/* .dirs
	g++ --std=c++14 -I cxxcompat/include cubesql/test_tokenizer.cpp cubesql/tokenizer.cpp -o ./bin/tests/test_tokenizer

bin/send: util/* proto/* send/* .dirs
	g++ --std=c++14 -I cxxcompat/include proto/proto.cpp send/send.cpp -o ./bin/send

bin/server: util/* proto/* server/* .dirs
	g++ --std=c++14 -I cxxcompat/include proto/proto.cpp server/server.cpp -o ./bin/server
