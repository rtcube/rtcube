#pragma once

#include <cstdlib>
#include <vector>

#define OP_NONE 0
#define OP_SUM 1
#define OP_MAX 2
#define OP_MIN 3
#define OP_AVG 4
#define OP_CNT 5

#define WHERE_NONE 0
#define WHERE_SET 1
#define WHERE_RANGE 2
#define WHERE_MAXRANGE 3

// Intermediate Representation
namespace IR
{
	struct Dim
	{
		uint range; // amount of different values

		inline Dim(uint range = 0): range(range) {}
	};

	struct Mea
	{
		enum Type: uint { Int, Float } type;

		inline Mea(Type type = Int): type(type) {}
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
		size_t num_rows;

		std::vector<int> dims; // dims of row1, then dims of row2, then ...
		std::vector<mea> meas; // meas of row1, then meas of row2, then ...

		Rows(size_t num_dims, size_t num_meas, size_t num_rows): num_dims(num_dims), num_meas(num_meas), num_rows(num_rows), dims(num_rows * num_dims), meas(num_rows * num_meas) {}

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

	struct Query
	{
		//Liczba wymiarów w kostce
		int DimCount;

		//Liczba miar w kostce
		int MeasCount;

		//Które wymiary selectujemy - tablica długości DimCount, ma 1 dla każdego wymiaru, który jest w select
		std::vector<int> selectDims;

		//Informacja o ograniczeniach dla wartości wymiaru - tablica długości DimCount z jedną z możliwych wartości
		//WHERE_NONE - brak ograniczeń
		//WHERE_SET - wartość wymiaru ze zbioru wartości (czyli IN {a, b, c, d})
		//WHERE_RANGE - wartość wymiaru z zakresu wartości (czyli IN <a, b> )
		//WHERE_MAXRANGE - wartość wymiaru z zakresu <0, dMAX>  (dMax - maksymalna wartość tego wymiaru). To ustawiamy dla każdego wymiaru, który jest w select (ma selectDims[d] == 1), ale
		//nie ma dla niego innych ograniczeń
		std::vector<int> whereDimMode;


		//Tablica długości DimCount, która dla każdego wymiaru, na którym są ograniczenia, trzyma informację ile jest wartości w ograniczeniu (w zakresie lub zbiorze)
		std::vector<int> whereDimValuesCounts;

		//Tablica, w której po kolei wpisane są zbiory wartości podane w select dla wymiarów
		std::vector<int> whereDimVals;

		//Tablica długości DimCount - dla każdego wymiaru mającego ograniczenie SET trzyma indeks początku jego zbioru w tablicy powyżej
		std::vector<int> whereDimValsStart;

		//Czyli mamy w query:
		//d0 in {0, 2, 4}
		//d2 in {1, 3, 5}
		// i mamy w kostce 3 wymiary. Wtedy:
		//whereDimVals = [0, 2, 4, 1, 3, 5]
		//whereDimValsStart = [0, 0, 3] - przy czym środkowe 0 (dla d1) jest nieistotne


		//Tutaj mamy dwie tablice długości DimCount. Dla każdego wymiaru, dla którego mamy WHERE_RANGE lub WHERE_MAXRANGE ustawiamy, whereStartRange[d] na minimalną wartość zakresu,
		//a whereEndRange[d] na maksymalną wartość zakresu + 1 (pierwszy poza). Dla pozostałych wymiarów wartość nie ma zanaczenia
		std::vector<int> whereStartRange;
		std::vector<int> whereEndRange;

		//Liczba operacji na miarach w select (np. select d1 d2 SUM(m1) AVG(m1) MIN(m2) MAX(m3) FROM ...  - tutaj mamy 4 operacje na miarach
		int operationsCount;

		//Tablica długości operationsCount. Dla każdej operacji wpisujemy tutaj jaka to ma być operacja
		std::vector<int> operationsTypes;

		//Tablica długości operationsCount. Dla każdej operacji wpisujemy tutaj na której mierze ma być wykonywana operacja
		std::vector<int> operationsMeasures;

		//Czyli operationTypes[i] == OP_MAX i operationMeasures[i] == 2 znaczy, że i-ta operacja to MAX(m2)

		Query(size_t dimCount, size_t measCount, size_t opCount) : DimCount(dimCount), MeasCount(measCount), selectDims(dimCount), whereDimMode(dimCount, WHERE_NONE), whereDimValuesCounts(dimCount, 0),
				whereDimValsStart(dimCount, 0), whereStartRange(dimCount, 0), whereEndRange(dimCount, 0), operationsCount(opCount), operationsTypes(opCount, OP_NONE), operationsMeasures(opCount)
		{}

		Query(){}


	};
	struct QueryResult
	{
		IR::Query query;

		//Liczba linii wyniku - ile jest możliwych kombinacji wartości wymiarów
		int resultsCount;

		//Rozmiar jednego wyniku - ile operacji na miarach było robionych
		int measPerResult;

		//Rozmiary kolejnych poziomów w tablicy resultMeas
		std::vector<int> selectDimSizes;

		//Wielowymiarowa tablica spłaszczona do jednowymiarowej. Zawiera kolejne linie wyniku. Jak się poruszać po tym -> GetQuerryResultString w RTQuery.cu
		std::vector<int> resultMeas;

		QueryResult(Query q)
		{
			query = q;

			selectDimSizes = std::vector<int>(q.DimCount);

			int currentSize = 1;
			for (int i = q.DimCount - 1; i >= 0; --i)
			{
				if (q.selectDims[i] != 0)
				{
					selectDimSizes[i] = currentSize;
					currentSize *= q.whereDimValuesCounts[i];
				}
			}

			resultsCount = currentSize;
			measPerResult = q.operationsCount;
			resultMeas = std::vector<int>(resultsCount * measPerResult, 0);
		}

		QueryResult(){}

	};
}
