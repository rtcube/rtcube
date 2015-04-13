#include "parser.h"
#include "tokenizer.h"

#include <iostream>
#include <cassert>

using namespace std;


int main(int argc, char** argv)
{
	auto tokens = CubeSQL::tokenize(R"(
SELECT
    5
WHERE
    yob < 1997
AND
    locality = "Warszawa"
ORDER BY
    5 DESC
LIMIT
    5)");

	auto q = CubeSQL::parse(tokens);

	assert(q.select.size() == 1);
	assert(dynamic_cast<CubeSQL::ConstantExpr*>(q.select[0].get()));
	assert(dynamic_cast<CubeSQL::ConstantExpr*>(q.select[0].get())->val == 5ll);

	assert(q.where.size() == 2);

	assert(q.where[0].field_name == "yob");
	assert(!q.where[0].array_specifier);
	assert(q.where[0].op == CubeSQL::Condition::LT);
	assert(q.where[0].a == 1997ll);

	assert(q.where[1].field_name == "locality");
	assert(!q.where[1].array_specifier);
	assert(q.where[1].op == CubeSQL::Condition::E);
	assert(q.where[1].a == string{"Warszawa"});

	assert(q.order_by.size() == 1);
	assert(dynamic_cast<CubeSQL::ConstantExpr*>(q.order_by[0].term.get())->val == 5ll);
	assert(!q.order_by[0].asc);

	assert(q.limit == 5);
	assert(q.offset == 0);

	cout << "OK" << endl;
}
