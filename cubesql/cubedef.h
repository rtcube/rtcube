#pragma once

#include <memory>
#include <vector>
#include <cxxcompat/optional>
#include "types.h"

namespace CubeSQL
{
	enum class ColType
	{
		None,
		Int,
		Float,
		IntRange,
		FloatRange,
		Set,
		Time,
		Text,
		Char
	};
	// Mea: only Int, Float, IntRange, FloatRange, Set.

	struct ColDef
	{
		std::string name;
		ColType type = ColType::None;
		int len = 1; // only for Int, Float, Char.
		AnySet s;
		AnyRange r;

		ColDef(std::string&& name = {}, ColType type = ColType::None, int len = 1): name{name}, type{type}, len(len) {}
		ColDef(const std::string& name, ColType type = ColType::None, int len = 1): name{name}, type{type}, len(len) {}
	};

	struct CubeDef
	{
		std::vector<ColDef> dims;
		std::vector<ColDef> meas;
	};
}
