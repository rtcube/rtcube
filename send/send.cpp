#include "../proto/proto.h"
#include "../util/HostPort.h"
#include <iostream>

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cerr << "Usage: " << argv[0] << " ip:port val1 val2 val3..." << std::endl;
		return 1;
	}

	auto dest = HostPort{argv[1]};

	auto V = std::vector<proto::value>{};
	for (auto i = 2; i < argc; ++i)
		V.emplace_back(argv[i]);

	auto msg = proto::serialize(V);

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

	auto sent = sendto(s, msg.data(), msg.size(), 0, (::sockaddr*) &sin6, sizeof(sin6));
	if (sent < 0)
	{
		perror("Sending packet.");
		return 1;
	}
	if (sent != msg.size())
	{
		std::cerr << "Invalid number of bytes sent - sent " << sent << ", expected " << msg.size() << std::endl;
		perror("Sending packet.");
		return 1;
	}

	return 0;
}
