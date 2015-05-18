#include "../proto/proto.h"
#include "../util/HostPort.h"
#include <iostream>
using namespace std;

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " ip:port" << std::endl;
		return 1;
	}

	auto dest = HostPort{argv[1]};

	::sockaddr_in6 sin6;
	::memset(&sin6, 0, sizeof(sin6));
	sin6.sin6_family = AF_INET6;
	::memcpy(&sin6.sin6_addr, dest.ip, 16);
	sin6.sin6_port = dest.port;

	auto s = socket(PF_INET6, SOCK_DGRAM, 0);
	if (s < 0)
	{
		perror("Opening socket.");
		return 1;
	}

	if (bind(s, (::sockaddr*) &sin6, sizeof(sin6)) < 0)
	{
		perror("Binding socket.");
		return 1;
	}

	for (;;)
	{
		char buffer[2048];
		socklen_t addr_len;

		auto len = recvfrom(s, buffer, 2048, 0, (::sockaddr*) &sin6, &addr_len);
		if (len <= 0)
		{
			perror("Receiving packet.");
			return 1;
		}

		auto msg = string{buffer, string::size_type(len)};
		auto V = proto::unserialize(msg);

		for (auto v : V)
			std::cout << v.data << " ";
		std::cout << std::endl;

		if (V.size() == 1 && V[0] == "DIE")
			break;
	}

	return 0;
}
