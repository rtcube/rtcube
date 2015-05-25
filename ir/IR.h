#pragma once

#include <cstdlib>
#include <vector>

// Intermediate Representation
namespace IR
{
	struct Dim
	{
		uint range; // amount of different values
	};

	struct Mea
	{
		enum Type: uint { Int, Float } type;
	};

	struct CubeDef
	{
		std::vector<Dim> dims;
		std::vector<Mea> meas;
	};

	union mea
	{
		int64_t i;
		double f;
	};

	struct Rows
	{
		size_t num_dims;
		size_t num_meas;

		std::vector<int> dims; // dims of row1, then dims of row2, then ...
		std::vector<mea> meas; // meas of row1, then meas of row2, then ...

		Rows(size_t num_dims, size_t num_meas, size_t num_rows): num_dims(num_dims), num_meas(num_meas), dims(num_rows * num_dims), meas(num_rows * num_meas) {}

		struct RowRef
		{
			Rows* r;
			size_t i;

			int* dims() {return r->dims.data() + i*r->num_dims;}
			mea* meas() {return r->meas.data() + i*r->num_meas;}

			const int* dims() const {return r->dims.data() + i*r->num_dims;}
			const mea* meas() const {return r->meas.data() + i*r->num_meas;}
		};

		struct ConstRowRef
		{
			const Rows* r;
			size_t i;

			const int* dims() const {return r->dims.data() + i*r->num_dims;}
			const mea* meas() const {return r->meas.data() + i*r->num_meas;}
		};

		ConstRowRef operator[](size_t i) const { return {this, i}; }
		RowRef operator[](size_t i) { return {this, i}; }
	};

	struct Query {};
	struct QueryResult {};
}
