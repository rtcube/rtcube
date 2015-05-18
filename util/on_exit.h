#pragma once

template <typename F>
struct OnExit
{
	bool act;
	F f;

	OnExit(F f): act{true}, f{f} {}

	OnExit(OnExit&& x): act{x.act}, f{x.f} {x.act = false;}
	OnExit(const OnExit&) = delete;

	void operator=(OnExit&& x) {act = x.act; f = x.f; x.act = false;}
	void operator=(const OnExit&) = delete;

	~OnExit() {if (act) f();}
};

template <typename F>
auto on_exit(F f) { return OnExit<F>{f}; }

#define ON_EXIT(v, f) auto v = on_exit([&](){f});
