#include "RTCube.cuh"
#include "RTQuery.cuh"
#include "RTUtil.cuh"
#include "RTCubeApi.h"

RTCube cube;

int *dimRanges;

void initCube()
{
	// Sample data, to be filled from cube definition
	float cardMemoryPartToFill = 0.2;
	int dimensionsCount = 6;
	int measuresCount = 3;
	dimRanges = (int*)malloc(dimensionsCount * sizeof(int));
	dimRanges[0] = 2000;
	dimRanges[1] = 5;
	dimRanges[2] = 6;
	dimRanges[3] = 3;
	dimRanges[4] = 5;
	dimRanges[5] = 30;

	cube = InitCube(cardMemoryPartToFill, dimensionsCount, dimRanges, measuresCount, 32, 32);

	printf("Cube created succesfully\n");
	PrintCubeInfo(cube);
}

void cubeInsert(int *row, int size)
{
	if (size != cube.DimensionsCount + cube.MeasuresCount)
	{
		std::cout << "Bad data size: " << size << " instead of " << cube.DimensionsCount << " dims + " << cube.MeasuresCount << " meas" << std::endl;
		return;
	}
	// For now account for only one row at a time
	int **dims;
	int **meas;
	int vectorsCount = 1;
	int dimCount = cube.DimensionsCount;
	int measCount = cube.MeasuresCount;
	dims = (int**)malloc(vectorsCount * sizeof(int*));
	meas = (int**)malloc(vectorsCount * sizeof(int*));
	for (int j = 0; j < vectorsCount; ++j)
	{
		dims[j] = (int*)malloc(dimCount * sizeof(int));
		meas[j] = (int*)malloc(measCount * sizeof(int));

		for (int i = 0; i < dimCount; ++i)
			dims[j][i] = row[i];

		for (int i = 0; i < measCount; ++i)
			meas[j][i] = row[dimCount + i];
	}
//	int measMax = 100;
//	GeneratePack(vectorsCount, dimCount, dimRanges, measCount, measMax, &dims, &meas);

	//PrintPack(vectorsCount, dimCount, measCount, dims, meas);

	thrust::device_ptr<int> DimensionsPacked;
	thrust::device_ptr<int> MeasuresPacked;

	PrepareDataForInsert(vectorsCount, dimCount, dims, measCount, meas, &DimensionsPacked, &MeasuresPacked);

	AddPack(cube, vectorsCount, DimensionsPacked, MeasuresPacked);

	thrust::device_free(DimensionsPacked);
	thrust::device_free(MeasuresPacked);

	FreePack(vectorsCount, &dims, &meas);

	//PrintCubeInfo(cube);
}

std::string cubeQuery()
{
	int operationsCount = 3;

	Querry q = InitQuerry(cube.DimensionsCount, cube.MeasuresCount, operationsCount);

	//Inicjalizujemy takie Query:
	//SELECT d0 d3 SUM(m0) CNT(m2) SUM(m2) FROM CUBE
	//WHERE d0 IN <0 19>
	//AND d1 IN{ 0 1 3 }
	//AND d2 IN{ 0 2 4 }

	//Oznaczamy 1 te wymiary wektora, które chcemy selectować (czyli z przykładu d0 i d3)
	q.d_SelectDims[0] = 1;
	q.d_SelectDims[3] = 1;

	//Tutaj pokazujemy jakie mają być limity w where
	q.d_WhereDimMode[0] = RTCUBE_WHERE_RANGE;	//d0 będzie brane z zakresu
	q.d_WhereDimMode[1] = RTCUBE_WHERE_SET;		//d1 będzie brane ze zbioru
	q.d_WhereDimMode[2] = RTCUBE_WHERE_SET;		//d2 będzie brane ze zbioru
	q.d_WhereDimMode[3] = RTCUBE_WHERE_MAXRANGE;	//d3 jest selectowane więc jeśli nie ma dla niego nic w where,
	//to trzeba ustawić _MAXRANGE

	//Tutaj dla każdego z wymiarów, dla którego mamy jakieś ograniczenie wpisujemy ile jest możliwych wartości
	q.d_WhereDimValuesCounts[0] = 20;	//ile wartości w zakresie
	q.d_WhereDimValuesCounts[1] = 3;	//ile wartości w ziorze
	q.d_WhereDimValuesCounts[2] = 3;	//ile wartości w zbiorze
	q.d_WhereDimValuesCounts[3] = dimRanges[3];	//ile wszystkich wartości dla wymiaru

	//Tutaj wpisujemy początki i końce zakresów
	q.d_WhereStartRange[0] = 0;
	q.d_WhereStartRange[3] = 0;

	q.d_WhereEndRange[0] = 19;
	q.d_WhereEndRange[3] = dimRanges[3];

	//Tutaj kolejno wpisujemy wszystkie zbiory wartości z where
	q.d_WhereDimVals = thrust::device_malloc<int>(6);
	q.d_WhereDimVals[0] = 0;
	q.d_WhereDimVals[1] = 1;
	q.d_WhereDimVals[2] = 3;
	q.d_WhereDimVals[3] = 0;
	q.d_WhereDimVals[4] = 2;
	q.d_WhereDimVals[5] = 4;

	//I teraz tutaj wpisujemy dla tych wymiarów, które mają wartości ze zbiorów, na którym indeksie w tablicy wyżej
	//zaczyna się zbiór dla danego wymiaru
	q.d_WhereDimValsStart[1] = 0;
	q.d_WhereDimValsStart[2] = 3;

	//Tutaj wpisujemy dla jakich miar chcemy wykonywać operacje
	q.OperationsMeasures[0] = 0;
	q.OperationsMeasures[1] = 2;
	q.OperationsMeasures[2] = 2;

	//Tutaj jakie to mają być operacje
	q.OperationsTypes[0] = RTCUBE_OP_SUM;
	q.OperationsTypes[1] = RTCUBE_OP_CNT;
	q.OperationsTypes[2] = RTCUBE_OP_SUM;

	PrintQuerry(q);

	//Tutaj zadajemy zapytanie do kostki
	QueryResult result = ProcessQuerry(cube, q);

	//I wypisujemy wynik
	//PrintQuerryResult(result);

	std::string resultString = GetQuerryResultString(result);
	std::cout << resultString << std::endl;

	//Teraz ten wynik trzeba przesłać na serwer zapytań (najpierw trzeba przekopiować do pamięci CPU)

	//Można zwolnić query i result
	FreeResult(result);
	FreeQuerry(q);

	return resultString;
}

void cubeStatus()
{
	PrintCubeInfo(cube);
}
