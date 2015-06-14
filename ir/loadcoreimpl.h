#include <string>

namespace IR
{
	class CoreImpl;

	using InitCore = auto (*)() -> CoreImpl*;

	auto loadCoreImpl(const std::string& type) -> IR::CoreImpl*;
}
