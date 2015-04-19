all: test_proto test_tokenizer test_parser bin/send bin/server bin/row_generator

test: test_proto test_tokenizer test_parser test_server

test_proto: bin/tests/test_proto
	./bin/tests/test_proto

test_tokenizer: bin/tests/test_tokenizer
	./bin/tests/test_tokenizer

test_parser: bin/tests/test_parser
	./bin/tests/test_parser

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

bin/tests/test_parser: util/* cubesql/* .dirs
	g++ --std=c++14 -I cxxcompat/include cubesql/test_parser.cpp cubesql/tokenizer.cpp cubesql/query.cpp cubesql/parser.cpp -o ./bin/tests/test_parser

bin/send: util/* proto/* send/* .dirs
	g++ --std=c++14 -I cxxcompat/include proto/proto.cpp send/send.cpp -o ./bin/send

bin/server: util/* proto/* server/* .dirs
	g++ --std=c++14 -I cxxcompat/include proto/proto.cpp server/server.cpp -o ./bin/server

bin/row_generator:	
	g++ --std=c++14 -I cxxcompat/include proto/proto.cpp row-generator/RowGenerator.cpp -o ./bin/row_generator
