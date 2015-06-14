#include "api.h"

#include <cassert>

void DummyCube::insert(const IR::Rows& rows)
{
	assert(rows.num_dims == def().dims.size());
	assert(rows.num_meas == def().meas.size());

	if (!cube.def.dims.size())
		return; // Cube was too big, we fail silently.

	for (int i = 0; i < rows.num_rows; ++i)
		std::copy(rows[i].meas(), rows[i].meas() + rows.num_meas, cube[rows[i].dims()]);
}

IR::QueryResult DummyCube::query(const IR::Query& q)
{
	auto rcube = resultCubeDef(def(), q);

	if (rcube.cube_size() == cube.def.cube_size())
		return cube.data; // Trololo.

	auto result = IR::QueryResult{};
	result.resize(rcube.cube_size());

	return result;
}
