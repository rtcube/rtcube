#include <iostream>
#include <vector>
#include "ir/core.h"

using namespace IR;
using namespace std;

void simpleTest(Core& core)
{
	auto def = CubeDef{};
	def.dims.emplace_back(5);
	def.dims.emplace_back(10);
	def.meas.emplace_back(Mea::Int);

	auto rows = Rows{2, 1, 1};
	rows[0].dims()[0] = 3;
	rows[0].dims()[1] = 7;
	rows[0].meas()[0].i = 55;

	auto db = core.make_db(def);
	db.insert(rows);
	auto result = db.query(Query{});
}

void fullTest(Core& core)
{
	auto def = CubeDef{};

	def.dims.push_back(10000);
	def.dims.push_back(100);
	def.dims.push_back(500);
	def.dims.push_back(10);
	def.dims.push_back(500);

	def.meas.push_back(IR::Mea::Float);
	def.meas.push_back(IR::Mea::Float);
	def.meas.push_back(IR::Mea::Float);

	IR::Rows rows = IR::Rows(5, 3, 10);

	for(int i = 0; i < 10; ++i)
	{
		for(int j = 0; j < 5; ++j)
		{
			rows.dims[i * 5 + j] = i + j;
		}

		for(int j = 0; j < 3; ++j)
		{
			IR::mea m;
			m.f = i + j + 0.5;
			rows.meas[i * 3 + j] = m;
		}
	}

	auto q = IR::Query{5};

	q.selectDims[0] = 1;
	q.selectDims[3] = 1;

	//Tutaj pokazujemy jakie mają być limity w where
	q.whereDimMode[0] = Query::CondType::Range;	//d0 będzie brane z zakresu
	q.whereDimMode[1] = Query::CondType::Set;		//d1 będzie brane ze zbioru
	q.whereDimMode[2] = Query::CondType::Set;		//d2 będzie brane ze zbioru
	q.whereDimMode[3] = Query::CondType::MaxRange;	//d3 jest selectowane więc jeśli nie ma dla niego nic w where,
													//to trzeba ustawić MaxRange

	//Tutaj dla każdego z wymiarów, dla którego mamy jakieś ograniczenie wpisujemy ile jest możliwych wartości
	q.whereDimValuesCounts[0] = 20;	//ile wartości w zakresie
	q.whereDimValuesCounts[1] = 3;	//ile wartości w ziorze
	q.whereDimValuesCounts[2] = 3;	//ile wartości w zbiorze
	q.whereDimValuesCounts[3] = 10;	//ile wszystkich wartości dla wymiaru

	//Tutaj wpisujemy początki i końce zakresów
	q.whereStartRange[0] = 0;
	q.whereStartRange[3] = 0;

	q.whereEndRange[0] = 19;
	q.whereEndRange[3] = 10;

	//Tutaj kolejno wpisujemy wszystkie zbiory wartości z where
	q.whereDimVals = vector<int>(6);
	q.whereDimVals[0] = 0;
	q.whereDimVals[1] = 1;
	q.whereDimVals[2] = 3;
	q.whereDimVals[3] = 0;
	q.whereDimVals[4] = 2;
	q.whereDimVals[5] = 4;

	//I teraz tutaj wpisujemy dla tych wymiarów, które mają wartości ze zbiorów, na którym indeksie w tablicy wyżej
	//zaczyna się zbiór dla danego wymiaru
	q.whereDimValsStart[1] = 0;
	q.whereDimValsStart[2] = 3;

	//Tutaj wpisujemy dla jakich miar chcemy wykonywać operacje, i jakie to operacje
	q.operationsMeasures.emplace_back(0);
	q.operationsTypes.emplace_back(Query::OperationType::Sum);

	q.operationsMeasures.emplace_back(2);
	q.operationsTypes.emplace_back(Query::OperationType::Cnt);

	q.operationsMeasures.emplace_back(2);
	q.operationsTypes.emplace_back(Query::OperationType::Sum);

	auto db = core.make_db(def);
	db.insert(rows);
	auto result = db.query(q);
}

int main(int argc, char** argv)
{
	if (argc != 2)
	{
		cerr << "Need core type." << endl;
		return -1;
	}

	auto core = Core{argv[1]};
	simpleTest(core);
	fullTest(core);
	cout << "OK" << endl;
}
