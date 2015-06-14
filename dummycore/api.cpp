#include "api.h"

#include <cassert>

void DummyCube::insert(const IR::Rows& rows)
{
	assert(rows.num_dims == def().dims.size());
	assert(rows.num_meas == def().meas.size());
}

IR::QueryResult DummyCube::query(const IR::Query& q)
{
	return IR::QueryResult(q);
}
