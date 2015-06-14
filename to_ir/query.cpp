#include "to_ir.h"

#include <unordered_map>

using CubeSQL::Range;
using CubeSQL::Set;
using CubeSQL::Int;
using CubeSQL::Float;
using CubeSQL::String;

struct ColInfo
{
	enum T { Dim, Mea } type;
	int id;
	int len;
};

void select_fields(const std::vector<CubeSQL::AnyExpr>& exprs, const std::unordered_map<std::string, ColInfo>& cols_by_name, IR::Query& ir)
{
	for (auto& expr : exprs)
	{
		if (auto f = dynamic_cast<CubeSQL::FieldNameExpr*>(expr.get()))
		{
			auto dim = cols_by_name.at(f->name);

			if (dim.type != ColInfo::Dim)
				continue;

			if (f->index)
				ir.selectDims[dim.id + *f->index - 1] = 1;
			else
				for (int i = 0; i < dim.len; ++i)
					ir.selectDims[dim.id + i] = 1;
		}
		else if (auto o = dynamic_cast<CubeSQL::OperationExpr*>(expr.get()))
			select_fields(o->args, cols_by_name, ir);
	}
}

auto toIRDimValue(const CubeSQL::ColDef& dim, const CubeSQL::AnyAtom& a) -> int
{
	switch (dim.type)
	{
		case CubeSQL::ColType::IntRange:
		{
			if (a.type != CubeSQL::Type::Int)
				throw std::invalid_argument("toIRDimValue");
			return dim.r.i.index_of(a.i);
		}
		break;

		case CubeSQL::ColType::Set:
			if (a.type != dim.s.type)
				throw std::invalid_argument("toIRDimValue");

			switch (dim.s.type)
			{
				case CubeSQL::Type::Int:
					return dim.s.i.index_of(a.i);

				case CubeSQL::Type::Float:
					return dim.s.f.index_of(a.f);

				case CubeSQL::Type::String:
					return dim.s.s.index_of(a.s);
			};
		break;

		case CubeSQL::ColType::Time:
			if (a.type != CubeSQL::Type::Int)
				throw std::invalid_argument("toIRDimValue");
			return a.i;
		break;

		case CubeSQL::ColType::Char:
			if (a.type != CubeSQL::Type::String)
				throw std::invalid_argument("toIRDimValue");
			if (a.s.size() != 1)
				throw std::invalid_argument("toIRDimValue");
			return a.s[0];
		break;

		default:
			throw std::invalid_argument("toIR");
	}
}

#include <iostream>
auto toIRIndex(const ColInfo& col, const std::optional<int>& index_sql)
{
	if (col.len == 1)
		return col.id;

	if (!index_sql) // Whole array!
		throw std::invalid_argument("toIRIndex");

	return col.id + *index_sql - 1;
}

auto toIR(const CubeSQL::CubeDef& cube_sql, const IR::CubeDef& cube_ir, const CubeSQL::Select& q) -> IR::Query
{
	auto ir = IR::Query{cube_ir.dims.size()};

	auto cols_by_name = std::unordered_map<std::string, ColInfo>{};

	{
		auto i = 0;
		for (const auto& col : cube_sql.dims)
		{
			cols_by_name[col.name] = {ColInfo::Dim, i, col.len};
			i += col.len;
		}
	}

	{
		auto i = 0;
		for (const auto& col : cube_sql.meas)
		{
			cols_by_name[col.name] = {ColInfo::Mea, i, col.len};
			i += col.len;
		}
	}

	select_fields(q.select, cols_by_name, ir);

	auto conds = std::vector<const CubeSQL::Condition*>(cube_ir.dims.size(), 0);
	for (auto& cond : q.where)
	{
		auto& dim = cols_by_name[cond.field_name];
		if (dim.type != ColInfo::Dim)
			continue;

		conds[toIRIndex(dim, cond.index)] = &cond;
	}

	auto i = 0;
	for (const auto& dim : cube_sql.dims)
	{
		for (auto j = 0; j < dim.len; ++j)
		{
			auto index = i + j;

			if (!conds[index])
			{
				if (ir.selectDims[i])
				{
					ir.whereDimMode[i] = IR::Query::CondType::MaxRange;
					ir.whereDimValsStart[i] = ir.whereDimVals.size();
					ir.whereDimValuesCounts[i] = 1;
					ir.whereDimVals.push_back(cube_ir.dims[i].range);
				}
				continue;
			}

			auto& cond = *conds[index];

			if (cond.op == CubeSQL::Condition::IN && cond.s)
			{
				ir.whereDimMode[index] = IR::Query::CondType::Set;
				auto copyValues = [&](const auto& set)
				{
					ir.whereDimValsStart[index] = ir.whereDimVals.size();
					ir.whereDimValuesCounts[index] = set.values.size();
					for (auto v : set.values)
						ir.whereDimVals.push_back(toIRDimValue(dim, v));
				};
				switch (cond.s.type)
				{
					case CubeSQL::Type::Int:
						copyValues(cond.s.i); break;

					case CubeSQL::Type::Float:
						copyValues(cond.s.f); break;

					case CubeSQL::Type::String:
						copyValues(cond.s.s); break;
				};
			}
			else if (cond.op == CubeSQL::Condition::E)
			{
				ir.whereDimMode[index] = IR::Query::CondType::Set;
				ir.whereDimValsStart[index] = ir.whereDimVals.size();
				ir.whereDimValuesCounts[index] = 1;
				ir.whereDimVals.push_back(toIRDimValue(dim, cond.a));
			}
			else if (cond.op == CubeSQL::Condition::IN && cond.r)
			{
				ir.whereDimMode[index] = IR::Query::CondType::Range;
				ir.whereDimValsStart[index] = ir.whereDimVals.size();
				ir.whereDimValuesCounts[index] = 2;
				auto copyValues = [&](const auto& range)
				{
					ir.whereDimVals.push_back(toIRDimValue(dim, range.left) + (range.left_inclusive ? 0 : 1));
					ir.whereDimVals.push_back(toIRDimValue(dim, range.right) - (range.right_inclusive ? 0 : 1));
				};
				switch (cond.s.type)
				{
					case CubeSQL::Type::Int:
						copyValues(cond.r.i); break;

					case CubeSQL::Type::Float:
						copyValues(cond.r.f); break;

					case CubeSQL::Type::String:
						copyValues(cond.r.s); break;
				};
			}
			else if (cond.op == CubeSQL::Condition::LT || cond.op == CubeSQL::Condition::GT || cond.op == CubeSQL::Condition::LTE || cond.op == CubeSQL::Condition::GTE)
			{
				auto left = (cond.op == CubeSQL::Condition::GT || cond.op == CubeSQL::Condition::GTE) ? cond.a : CubeSQL::AnyAtom{};
				if (cond.op == CubeSQL::Condition::GT)
				{
					switch (left.type)
					{
						case CubeSQL::Type::Int:
							++left.i; break;

						case CubeSQL::Type::Float:
							++left.f; break;

						case CubeSQL::Type::String:
							throw std::invalid_argument("toIR");
					};
				}

				auto right = (cond.op == CubeSQL::Condition::LT || cond.op == CubeSQL::Condition::LTE) ? cond.a : CubeSQL::AnyAtom{};
				if (cond.op == CubeSQL::Condition::LT)
				{
					switch (right.type)
					{
						case CubeSQL::Type::Int:
							--right.i; break;

						case CubeSQL::Type::Float:
							--right.f; break;

						case CubeSQL::Type::String:
							throw std::invalid_argument("toIR");
					};
				}

				auto irleft = left ? toIRDimValue(dim, left) : 0;
				auto irright = right ? toIRDimValue(dim, right) : cube_ir.dims[index].range;

				ir.whereDimMode[index] = IR::Query::CondType::Range;
				ir.whereDimValsStart[index] = ir.whereDimVals.size();
				ir.whereDimValuesCounts[index] = 2;
				ir.whereDimVals.push_back(irleft);
				ir.whereDimVals.push_back(irright);
			}
			else
				throw std::invalid_argument("toIR");
		}

		i += dim.len;
	}

	// Operations
	for (auto& expr : q.select)
	{
		auto o = dynamic_cast<CubeSQL::OperationExpr*>(expr.get());
		if (!o)
			continue;

		if (o->args.size() != 1)
			continue;

		auto& arg = o->args[0];
		auto f = dynamic_cast<CubeSQL::FieldNameExpr*>(arg.get());
		if (!o)
			continue;

		auto mea = cols_by_name.at(f->name);

		if (mea.type != ColInfo::Mea)
			continue;

		auto op =
			(o->name == "SUM") ? IR::Query::OperationType::Sum :
			(o->name == "MAX") ? IR::Query::OperationType::Max :
			(o->name == "MIN") ? IR::Query::OperationType::Min :
			(o->name == "AVG") ? IR::Query::OperationType::Avg :
			(o->name == "COUNT") ? IR::Query::OperationType::Cnt :
			IR::Query::OperationType::None;

		ir.operationsMeasures.push_back(toIRIndex(mea, f->index));
		ir.operationsTypes.push_back(op);
	}

	return ir;
}
