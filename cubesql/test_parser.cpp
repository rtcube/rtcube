#include "parser.h"
#include "tokenizer.h"

#include <iostream>
#include <cassert>

using namespace std;


void testSelect()
{
	auto q = CubeSQL::parse(R"(
SELECT
    locality, yob, SUM(signatures)
WHERE
    yob IN <1980, 1997>
AND
    locality IN {"Warszawa", "Kraków"}
AND
    pesel[10] IN {1, 3, 5, 7, 9}
ORDER BY
    SUM(signatures) DESC
LIMIT
    5)");

	{
		assert(q.select.size() == 3);

		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(q.select[0].get()));
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(q.select[0].get())->name == "locality");
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(q.select[0].get())->index == nullopt);

		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(q.select[1].get()));
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(q.select[1].get())->name == "yob");
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(q.select[1].get())->index == nullopt);

		assert(dynamic_cast<CubeSQL::OperationExpr*>(q.select[2].get()));
		auto sum = dynamic_cast<CubeSQL::OperationExpr*>(q.select[2].get());
		assert(sum->name == "SUM");
		assert(sum->args.size() == 1);
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(sum->args[0].get()));
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(sum->args[0].get())->name == "signatures");
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(sum->args[0].get())->index == nullopt);
	}

	{
		assert(q.where.size() == 3);

		assert(q.where[0].field_name == "yob");
		assert(!q.where[0].index);
		assert(q.where[0].op == CubeSQL::Condition::IN);
		assert(q.where[0].r == CubeSQL::Range<CubeSQL::Int>(true, true, 1980, 1997));

		assert(q.where[1].field_name == "locality");
		assert(!q.where[1].index);
		assert(q.where[1].op == CubeSQL::Condition::IN);
		auto WK = CubeSQL::Set<CubeSQL::String>{string{"Warszawa"}, string{"Kraków"}};
		assert(q.where[1].s == WK);

		assert(q.where[2].field_name == "pesel");
		assert(q.where[2].index);
		assert(*q.where[2].index == 10ll);
		assert(q.where[2].op == CubeSQL::Condition::IN);
		auto s = CubeSQL::Set<CubeSQL::Int>{1, 3, 5, 7, 9};
		assert(q.where[2].s == s);
	}

	{
		assert(q.order_by.size() == 1);
		auto sum = dynamic_cast<CubeSQL::OperationExpr*>(q.order_by[0].term.get());
		assert(sum->name == "SUM");
		assert(sum->args.size() == 1);
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(sum->args[0].get()));
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(sum->args[0].get())->name == "signatures");
		assert(dynamic_cast<CubeSQL::FieldNameExpr*>(sum->args[0].get())->index == nullopt);
		assert(!q.order_by[0].asc);
	}

	assert(q.limit == 5);
	assert(q.offset == 0);

	cout << "OK" << endl;
}

void testCubeDef()
{
	auto cube = CubeSQL::parseCubeDef(R"(
DIM recv TIME;

DIM yob <1900,2015>;
DIM commune CHAR[7];
DIM locality text;
DIM pesel CHAR[11];

MEA signatures <0,1000000>;
MEA applications <0,100000>.)");

	assert(cube.dims.size() == 5);
	assert(cube.meas.size() == 2);

	assert(cube.dims[0].name == "recv");
	assert(cube.dims[0].type == CubeSQL::ColType::Time);
	assert(cube.dims[0].len == 1);

	assert(cube.dims[1].name == "yob");
	assert(cube.dims[1].type == CubeSQL::ColType::IntRange);
	assert(cube.dims[1].len == 1);
	assert(cube.dims[1].r == CubeSQL::Range<CubeSQL::Int>(true, true, 1900, 2015));

	assert(cube.dims[2].name == "commune");
	assert(cube.dims[2].type == CubeSQL::ColType::Char);
	assert(cube.dims[2].len == 7);

	assert(cube.dims[3].name == "locality");
	assert(cube.dims[3].type == CubeSQL::ColType::Text);
	assert(cube.dims[3].len == 1);

	assert(cube.dims[4].name == "pesel");
	assert(cube.dims[4].type == CubeSQL::ColType::Char);
	assert(cube.dims[4].len == 11);

	assert(cube.meas[0].name == "signatures");
	assert(cube.meas[0].type == CubeSQL::ColType::IntRange);
	assert(cube.meas[0].len == 1);
	assert(cube.meas[0].r == CubeSQL::Range<CubeSQL::Int>(true, true, 0, 1000000));

	assert(cube.meas[1].name == "applications");
	assert(cube.meas[1].type == CubeSQL::ColType::IntRange);
	assert(cube.meas[1].len == 1);
	assert(cube.meas[1].r == CubeSQL::Range<CubeSQL::Int>(true, true, 0, 100000));
}

int main(int argc, char** argv)
{
	testSelect();
	testCubeDef();
}
