#pragma once

#include <memory>
#include <vector>
#include <cxxcompat/optional>
#include "types.h"

namespace CubeSQL
{
	struct Expr
	{
		virtual ~Expr();
	};

	using AnyExpr = std::unique_ptr<Expr>;

	struct FieldNameExpr: public Expr
	{
		std::string field_name;
		std::optional<int> array_specifier;
	};

	struct ConstantExpr: public Expr
	{
		AnyAtom val;
		ConstantExpr(AnyAtom&& v): val(std::move(v)) {}
		ConstantExpr(const AnyAtom& v): val(v) {}
	};

	struct OperationExpr : public Expr
	{
		std::string operation_name;
		std::vector<AnyExpr> args;
	};

	struct Condition
	{
		std::string field_name;
		std::optional<int> array_specifier;
		enum Operator
		{
			E,
			LT,
			GT,
			LTE,
			GTE,
			NE,
			IN
		} op;

		AnyAtom  a;
		AnyRange r;
		AnySet   s;
	};

	struct OrderingTerm
	{
		AnyExpr term;
		bool asc = true;

		OrderingTerm(AnyExpr term, bool asc = true): term(std::move(term)), asc(asc) {}
	};

	struct Select
	{
		std::vector<AnyExpr> select;
		std::vector<Condition> where;
		std::vector<OrderingTerm> order_by;
		std::optional<int> limit;
		int offset = 0;
	};
}
