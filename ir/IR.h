#pragma once

#include <cstdlib>
#include <cstdint>
#include <cassert>
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

		uint64_t cube_size() const
		{
			uint64_t cube_size = 1;
			for (const Dim& dim : dims)
				cube_size *= dim.range;
			cube_size *= meas.size();
			return cube_size;
		}
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

		std::vector<uint> dims; // dims of row1, then dims of row2, then ...
		std::vector<mea> meas; // meas of row1, then meas of row2, then ...

		Rows(size_t num_dims, size_t num_meas, size_t num_rows): num_dims(num_dims), num_meas(num_meas), num_rows(num_rows), dims(num_rows * num_dims), meas(num_rows * num_meas) {}

		struct RowRef
		{
			Rows* r;
			size_t i;

			uint* dims() {return r->dims.data() + i*r->num_dims;}
			mea* meas() {return r->meas.data() + i*r->num_meas;}

			const uint* dims() const {return r->dims.data() + i*r->num_dims;}
			const mea* meas() const {return r->meas.data() + i*r->num_meas;}
		};

		struct ConstRowRef
		{
			const Rows* r;
			size_t i;

			const uint* dims() const {return r->dims.data() + i*r->num_dims;}
			const mea* meas() const {return r->meas.data() + i*r->num_meas;}
		};

		ConstRowRef operator[](size_t i) const { return {this, i}; }
		RowRef operator[](size_t i) { return {this, i}; }
	};

	struct Query
	{
		#if __cplusplus >= 201103L
		enum class OperationType : int
		{
			None = 0,
			Sum = 1,
			Max = 2,
			Min = 3,
			Avg = 4,
			Cnt = 5
		};

		enum class CondType : int
		{
			None = 0,
			Set = 1,
			Range = 2,
			MaxRange = 3,
		};
		#else
		typedef int OperationType;
		typedef int CondType;
		#endif

		//Które wymiary selectujemy - tablica długości DimCount, ma 1 dla każdego wymiaru, który jest w select
		std::vector<int> selectDims;

		//Informacja o ograniczeniach dla wartości wymiaru - tablica długości DimCount z jedną z możliwych wartości
		//WHERE_NONE - brak ograniczeń
		//WHERE_SET - wartość wymiaru ze zbioru wartości (czyli IN {a, b, c, d})
		//WHERE_RANGE - wartość wymiaru z zakresu wartości (czyli IN <a, b> )
		//WHERE_MAXRANGE - wartość wymiaru z zakresu <0, dMAX>  (dMax - maksymalna wartość tego wymiaru). To ustawiamy dla każdego wymiaru, który jest w select (ma selectDims[d] == 1), ale
		//nie ma dla niego innych ograniczeń
		std::vector<CondType> whereDimMode;

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

		//Tablica długości operationsCount. Dla każdej operacji wpisujemy tutaj jaka to ma być operacja
		std::vector<OperationType> operationsTypes;

		//Tablica długości operationsCount. Dla każdej operacji wpisujemy tutaj na której mierze ma być wykonywana operacja
		std::vector<int> operationsMeasures;

		//Czyli operationTypes[i] == OP_MAX i operationMeasures[i] == 2 znaczy, że i-ta operacja to MAX(m2)

		Query(size_t dimCount = 0):
			selectDims(dimCount),
#if __cplusplus >= 201103L
			whereDimMode(dimCount, CondType::None),
#else
			whereDimMode(dimCount, WHERE_NONE),
#endif
			whereDimValuesCounts(dimCount, 0),
			whereDimValsStart(dimCount, 0),
			whereStartRange(dimCount, 0),
			whereEndRange(dimCount, 0)
		{}
	};

	struct Cube
	{
		CubeDef def;
		std::vector<mea> data;

		Cube(const CubeDef& def)
			: def(def)
		{
			data.resize(def.cube_size());
		}

		Cube(const CubeDef& def, const std::vector<mea>& data)
			: def(def)
			, data(data)
		{
			assert(data.size() == def.cube_size());
		}

		// Invariant: len(indexes) == def.dims.size()
		mea* operator[](const uint64_t* indexes)
		{
			return &data[encode_index(indexes)];
		}

		// Invariant: len(indexes) == def.dims.size()
		mea* operator[](const uint* indexes)
		{
			return &data[encode_index(indexes)];
		}

		// Invariant: len(indexes) == def.dims.size()
		template <typename T>
		uint64_t encode_index(const T* indexes)
		{
			uint64_t index = indexes[0];

			for (int i = 1; i < def.dims.size(); ++i)
				index = index * def.dims[i].range + indexes[i];

			index *= def.meas.size();

			return index;
		}

		// Invariant: len(indexes) == def.dims.size()
		// Post-con: RV == indexes.
		template <typename T>
		T* decode_index(uint64_t index, T* indexes)
		{
			index /= def.meas.size();

			for (int i = def.dims.size() - 1; i > 0; --i)
			{
				indexes[i] = index % def.dims[i].range;
				index /= def.dims[i].range;
			}

			indexes[0] = index;

			return indexes;
		}
	};

	inline CubeDef resultCubeDef(const CubeDef& c, const Query& q)
	{
		CubeDef r;

		for (int i = 0; i < q.selectDims.size(); ++i)
			if (q.selectDims[i])
				r.dims.push_back(q.whereDimValuesCounts[i]);

		for (int i = 0; i < q.operationsMeasures.size(); ++i)
			r.meas.push_back(q.operationsTypes[i] == Query::OperationType::Cnt ? Mea::Int : c.meas[i]);

		return r;
	}

	typedef std::vector<mea> QueryResult;
}
