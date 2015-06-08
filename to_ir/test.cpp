#include "../cubesql/parser.h"
#include "../cubesql/tokenizer.h"
#include "to_ir.h"

#include <iostream>
#include <cassert>

using namespace std;


void testCubeDef()
{
	auto cube = CubeSQL::parseCubeDef(R"(
dim recv TIME,

dim yob <1900,2015>,
dim commune CHAR[7],
dim pesel <0,9>[11],

mea signatures <0,1000000>,
mea applications <0,100000>)");

	auto ir = toIR(cube);
	assert(ir.dims.size() == 1 + 1 + 7 + 11);
	assert(ir.dims[0].range == 0);
	assert(ir.dims[1].range == 2015-1900+1);
	for (int i = 2; i < 2 + 7; ++i)
		assert(ir.dims[i].range == 256);
	for (int i = 2 + 7; i < 2 + 7 + 11; ++i)
		assert(ir.dims[i].range == 10);

	assert(ir.meas.size() == 2);
	assert(ir.meas[0].type == IR::Mea::Int);
	assert(ir.meas[1].type == IR::Mea::Int);
}


void testRows()
{
	auto cube = CubeSQL::parseCubeDef(R"(
dim recv TIME,

dim yob <1900,2015>,
dim commune CHAR[7],
dim pesel <0,9>[11],

mea signatures <0,1000000>,
mea applications <0,100000>)");
	auto cube_ir = toIR(cube);

	auto data = std::vector<proto::value>{
		10,
		1915,
		"1465011",
		"92929200001",
		50,
		100
	};
	auto data_ir = toIR(cube, cube_ir, {data});

	assert(data_ir.num_dims == 1 + 1 + 7 + 11);
	assert(data_ir.num_meas == 2);

	assert(data_ir.dims.size() == data_ir.num_dims);
	assert(data_ir.meas.size() == data_ir.num_meas);

	assert(data_ir[0].dims()[ 0] == 10);
	assert(data_ir[0].dims()[ 1] == 1915-1900);
	assert(data_ir[0].dims()[ 2] == '1');
	assert(data_ir[0].dims()[ 3] == '4');
	assert(data_ir[0].dims()[ 4] == '6');
	assert(data_ir[0].dims()[ 5] == '5');
	assert(data_ir[0].dims()[ 6] == '0');
	assert(data_ir[0].dims()[ 7] == '1');
	assert(data_ir[0].dims()[ 8] == '1');
	assert(data_ir[0].dims()[ 9] == 9);
	assert(data_ir[0].dims()[10] == 2);
	assert(data_ir[0].dims()[11] == 9);
	assert(data_ir[0].dims()[12] == 2);
	assert(data_ir[0].dims()[13] == 9);
	assert(data_ir[0].dims()[14] == 2);
	assert(data_ir[0].dims()[15] == 0);
	assert(data_ir[0].dims()[16] == 0);
	assert(data_ir[0].dims()[17] == 0);
	assert(data_ir[0].dims()[18] == 0);
	assert(data_ir[0].dims()[19] == 1);

	assert(data_ir[0].meas()[ 0].i == 50);
	assert(data_ir[0].meas()[ 1].i == 100);
}

void testQuery()
{
	auto cube = CubeSQL::parseCubeDef(R"(
dim recv <0,100000>,

dim yob <1900,2015>,
dim commune CHAR[7],
dim pesel <0,9>[11],

mea signatures <0,1000000>,
mea applications <0,100000>)");
	auto cube_ir = toIR(cube);

	auto query = CubeSQL::parse(R"(
SELECT recv, yob
WHERE pesel[10] in {1, 3, 5, 7, 9}
)");
	auto query_ir = toIR(cube, cube_ir, query);

	assert(query_ir.DimCount == cube_ir.dims.size());
	assert(query_ir.MeasCount == cube_ir.meas.size());

	assert(query_ir.selectDims[0] == 1);
	assert(query_ir.selectDims[1] == 1);
	for (int i = 2; i < cube_ir.dims.size(); ++i)
		assert(query_ir.selectDims[i] == 0);

	int f = 0;
	int v = 0;

	assert(query_ir.whereDimMode[f] == 3);
	assert(query_ir.whereDimValsStart[f] == v);
	assert(query_ir.whereDimValuesCounts[f] == 1);
	assert(query_ir.whereDimVals[v++] == 100001);
	++f;

	assert(query_ir.whereDimMode[f] == 3);
	assert(query_ir.whereDimValsStart[f] == v);
	assert(query_ir.whereDimValuesCounts[f] == 1);
	assert(query_ir.whereDimVals[v++] == 2015-1900+1);
	++f;

	// commune[1-7], pesel[1-9]
	for (; f < 1 + 1 + 7 + 9; ++f)
	{
		assert(query_ir.whereDimMode[f] == 0);
		assert(query_ir.whereDimValsStart[f] == 0);
		assert(query_ir.whereDimValuesCounts[f] == 0);
	}

	// pesel[10]
	assert(query_ir.whereDimMode[f] == 1);
	assert(query_ir.whereDimValsStart[f] == v);
	assert(query_ir.whereDimValuesCounts[f] == 5);
	assert(query_ir.whereDimVals[v++] == 1);
	assert(query_ir.whereDimVals[v++] == 3);
	assert(query_ir.whereDimVals[v++] == 5);
	assert(query_ir.whereDimVals[v++] == 7);
	assert(query_ir.whereDimVals[v++] == 9);
	++f;

	// pesel[11]
	assert(query_ir.whereDimMode[f] == 0);
	assert(query_ir.whereDimValsStart[f] == 0);
	assert(query_ir.whereDimValuesCounts[f] == 0);
	++f;
}

int main(int argc, char** argv)
{
	testCubeDef();
	testRows();
	testQuery();
	cout << "OK" << endl;
}
