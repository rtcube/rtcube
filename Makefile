test: test_proto

test_proto: tests/test_proto
	./tests/test_proto

tests:
	mkdir tests

tests/test_proto: util/* proto/* tests
	g++ --std=c++14 -I cxxcompat/include proto/test.cpp proto/proto.cpp -o ./tests/test_proto
