#include "parser.h"
#include "tokenizer.h"

#include <iostream>
#include <cassert>

using namespace std;


int main(int argc, char** argv)
{
	auto tokens = CubeSQL::tokenize(R"(
SELECT
    locality, yob, SUM(signatures)
WHERE
    yob IN <1980, 1997>
AND
    locality = "Warszawa"
ORDER BY
    SUM(signatures) DESC
LIMIT
    5)");

	auto q = CubeSQL::parse(tokens);

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
		assert(q.where.size() == 2);

		assert(q.where[0].field_name == "yob");
		assert(!q.where[0].array_specifier);
		assert(q.where[0].op == CubeSQL::Condition::IN);
		assert(q.where[0].r == CubeSQL::Range<CubeSQL::Int>(true, true, 1980, 1997));

		assert(q.where[1].field_name == "locality");
		assert(!q.where[1].array_specifier);
		assert(q.where[1].op == CubeSQL::Condition::E);
		assert(q.where[1].a == string{"Warszawa"});
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
