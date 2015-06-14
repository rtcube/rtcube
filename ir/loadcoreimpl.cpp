#include "loadcoreimpl.h"

#include <dlfcn.h>
#include <stdexcept>

namespace IR
{
	auto loadCoreImpl(const std::string& type) -> IR::CoreImpl*
	{
		auto lib = dlopen(("librt" + type + "core.so").c_str(), RTLD_LAZY);

		if (!lib)
			throw std::runtime_error{"loadCoreImpl(" + type + "): no library"};

		auto init = dlsym(lib, "init_core");

		if (!init)
			throw std::runtime_error{"loadCoreImpl(" + type + "): no symbol"};

		auto init_f = InitCore(init);
		return init_f();
	}
}
