#include "proto.h"
#include <cassert>
#include <iostream>
#include <sstream>

using namespace std;

void testParse()
{
	auto elems = proto::unserialize(string(1, char(5)) + "hello");
	assert(elems.size() == 1);
	assert(elems[0] == string("hello"));
}

void testParseMulti()
{
	auto elems = proto::unserialize(string(1, char(5)) + "hello" + string(1, char(1)) + "W");
	assert(elems.size() == 2);
	assert(elems[0] == string("hello"));
	assert(elems[1] == string("W"));
}

void testRead()
{
	auto input = string(1, char(5)) + "hello" + string(1, char(1)) + "W";
	stringstream ss{input};
	auto v1 = proto::read(ss);
	assert(bool(v1));
	assert(*v1 == string("hello"));
	auto v2 = proto::read(ss);
	assert(bool(v2));
	assert(*v2 == string("W"));
	auto v3 = proto::read(ss);
	assert(!bool(v3));
}

void testSerialize()
{
	auto elems = vector<string>{"hello", "W"};
	auto output = proto::serialize(elems);
	assert(output == string(1, char(5)) + "hello" + string(1, char(1)) + "W");
}

void testWrite()
{
	stringstream ss;
	proto::write(ss, "hello");
	proto::write(ss, "W");
	assert(ss.str() == string(1, char(5)) + "hello" + string(1, char(1)) + "W");
}

void testReal()
{
	auto data = vector<string>{"Janusz", "1992", "2015-03-01", "1000", "50"};
	auto s = proto::serialize(data);
	auto d = proto::unserialize(s);
	assert(d.size() == data.size());
	for (auto i = 0; i < data.size(); ++i)
		assert(d[i] == data[i]);
}

void testRealInt()
{
	auto data = vector<proto::value>{"Janusz", 1992, "2015-03-01", 1000, 50};
	auto s = proto::serialize(data);
	auto d = proto::unserialize(s);
	assert(d.size() == data.size());
	for (auto i = 0; i < data.size(); ++i)
		assert(d[i] == data[i]);
	assert(int(d[1]) == 1992);

	try
	{
		std::cout << int(d[0]) << std::endl; // cout is required, so compiler won't optimize int() away.
		assert(false);
	}
	catch (std::domain_error&) {}
}

void testCoolSerialize()
{
	auto data = vector<proto::value>{"Janusz", 1992, "2015-03-01", 1000, 50};
	auto s = proto::serialize("Janusz", 1992, "2015-03-01", 1000, 50);
	auto d = proto::unserialize(s);
	assert(d.size() == data.size());
	for (auto i = 0; i < data.size(); ++i)
		assert(d[i] == data[i]);
}

template <typename T>
auto my_tuple_size(T) -> size_t
{
	return std::tuple_size<T>();
}

void testCoolUnserialize()
{
	auto data = vector<proto::value>{"Janusz", 1992, "2015-03-01", 1000, 50};
	auto s = proto::serialize(data);
	auto d = proto::unserialize<string, int, string, int, int>(s);
	assert(my_tuple_size(d) == data.size());
	assert(get<0>(d) == "Janusz");
	assert(get<1>(d) == 1992);
	assert(get<2>(d) == "2015-03-01");
	assert(get<3>(d) == 1000);
	assert(get<4>(d) == 50);
}

void testCoolReal()
{
	auto s = proto::serialize("Janusz", 1992, "2015-03-01", 1000, 50);
	string name;
	int yob;
	string regdate;
	int signatures;
	int friends;
	tie(name, yob, regdate, signatures, friends) = proto::unserialize<string, int, string, int, int>(s);
	assert(name == "Janusz");
	assert(yob == 1992);
	assert(regdate == "2015-03-01");
	assert(signatures == 1000);
	assert(friends == 50);
}

void testEvenCooler()
{
	auto s = proto::serialize("Janusz", 1992, "2015-03-01", 1000, 50);
	auto d = proto::unserialize(s);
	assert(d[0] == "Janusz");
	assert(d[1] == 1992);
	assert(d[2] == "2015-03-01");
	assert(d[3] == 1000);
	assert(d[4] == 50);
}

int main(int argc, char** argv)
{
	testParse();
	testParseMulti();
	testRead();
	testSerialize();
	testWrite();
	testReal();
	testRealInt();
	testCoolSerialize();
	testCoolUnserialize();
	testCoolReal();
	testEvenCooler();
	cout << "OK" << endl;
}
