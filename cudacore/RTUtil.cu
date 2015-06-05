#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "api.h"
#include "RTCube.cuh"
#include "RTUtil.cuh"

void PrintVector(int dimCount, int *dimVals, int measCount, int *measVals)
{
	for(int i = 0; i < dimCount; ++i)
		printf("%7d", dimVals[i]);
	printf("   |");
	for(int i = 0; i < measCount; ++i)
		printf("%4d", measVals[i]);
	printf("\n");
}

void PrintPack(int vecCount, int dimCount, int measCount, int **dims, int **meas)
{
	for(int i = 0; i < vecCount; ++i)
		PrintVector(dimCount, dims[i], measCount, meas[i]);
	printf("\n");
}

void GeneretVector(int dimCount, int *dimRanges, int measCount, int measMax, int **dimVals, int **measVals)
{
	*dimVals = (int*)malloc(dimCount * sizeof(int));
	*measVals = (int*)malloc(measCount * sizeof(int));

	for(int i = 0; i < dimCount; ++i)
		(*dimVals)[i] = rand() % dimRanges[i];

	for(int i = 0; i < measCount; ++i)
		(*measVals)[i] = rand() % measMax;
}

void GeneratePack(int vecCount, int dimCount, int *dimRanges, int measCount, int measMax, int ***dims, int ***meas)
{
	*dims = (int**)malloc(vecCount * sizeof(int*));
	*meas = (int**)malloc(vecCount * sizeof(int*));

	for(int i = 0; i < vecCount; ++i)
		GeneretVector(dimCount, dimRanges, measCount, measMax, &((*dims)[i]), &((*meas)[i]));
}

void FreePack(int vecCount, int ***dims, int ***meas)
{
	for(int i = 0; i < vecCount; ++i)
	{
		free((*dims)[i]);
		free((*meas)[i]);
	}

	free(*dims);
	free(*meas);
}

void PrintPackedPack(int vecCount, int dimCount, int *dims, int measCount, int *meas)
{
	for(int i = 0; i < vecCount; ++i)
	{
		for(int j = 0; j < dimCount; ++j)
			printf("%7d", dims[j * vecCount + i]);
		printf("   |");
		for(int j = 0; j < measCount; ++j)
			printf("%4d", meas[j * vecCount + i]);
		printf("\n");
	}
	printf("\n");
}

void PrepareDataForInsert(int vecCount, int dimCount, int **dims, int measCount, int **meas, thrust::device_ptr<int> *d_dimsPacked, thrust::device_ptr<int> *d_measPacked)
{
	int *h_dimsPacked = (int*)malloc(vecCount * dimCount * sizeof(int));
	int *h_measPacked = (int*)malloc(vecCount * measCount * sizeof(int));

	for(int i = 0; i < vecCount; ++i)
	{
		for(int j = 0; j < dimCount; ++j)
			h_dimsPacked[j * vecCount + i] = dims[i][j];

		for(int j = 0; j < measCount; ++j)
			h_measPacked[j * vecCount + i] = meas[i][j];
	}

	//PrintPackedPack(vecCount, dimCount, h_dimsPacked, measCount, h_measPacked);

	*d_dimsPacked = thrust::device_malloc<int>(vecCount * dimCount);
	thrust::copy(h_dimsPacked, h_dimsPacked + vecCount * dimCount, *d_dimsPacked);

	*d_measPacked = thrust::device_malloc<int>(vecCount * measCount);
	thrust::copy(h_measPacked, h_measPacked + vecCount * measCount, *d_measPacked);

	free(h_dimsPacked);
	free(h_measPacked);
}

void RunSample()
{
	//Wygenerowanie przykładowych danych - normalnie trzeba je odebrać z sieci
	int vectorsCount = 200000;
	int dimensionsCount = 6;
	int measuresCount = 3;

	int *dimRanges = (int*)malloc(dimensionsCount * sizeof(int));
	dimRanges[0] = 2000;
	dimRanges[1] = 5;
	dimRanges[2] = 6;
	dimRanges[3] = 3;
	dimRanges[4] = 5;
	dimRanges[5] = 4;

	int measMax = 100;

	int **dims;
	int **meas;

	GeneratePack(vectorsCount, dimensionsCount, dimRanges, measuresCount, measMax, &dims, &meas);
	//PrintPack(vectorsCount, dimensionsCount, measuresCount, dims, meas);

	//Inicjalizacja kostki
	float cardMemoryPartToFill = 0.2;	//Znaczy, że na kostkę zajmujemy 20% pamięci karty

	//Funkcją InitCube inicjalizujemy kostkę, czyli budujemy strukturę RTCube, która trzyma wszystko co potrzebne i jest parametrem funkcji dodawania danych i odpytywania kostki
	//Parametry:
	//cardMemoryPartToFill - double z zakresu 0-1, który procentowo mówi ile pamięci karty ma zająć tworzona kostka, liczymy tylko wolną pamięć w chwili tworzenia kostki
	//(np. 0.5 oznacza zajęcie aktualnie wolnej pamięci w 50%)
	//dimensionsCount - liczba wymiarów kostki
	//dimRanges - tablica długości dimensionsCount, która mówi, jakie są maksymalne zakresy wartości wymiarów (przyjmujemy minimalne zawsze równe 0). Czyli dla 3 wymiarów ta tablica
	//to: {10, 3, 20} i to znaczy, że pierwszy wymiar ma wartości od 0 do 9, drugi od 0 do 2 itd.
	//measuresCount - liczba miar trzymanych w kostce
	//blocks - liczba bloków używanych do operacji na kostce
	//threads - liczba wątków -||-
	RTCube cube = InitCube(cardMemoryPartToFill, dimensionsCount, dimRanges, measuresCount, 32, 32);

	//Testowo wypisujemy ogólne informacje o kostce
	printf("Cube created succesfully\n");
	PrintCubeInfo(cube);
	printf("\n\n");

	cudaEvent_t start, stop;
	float time;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//Preprocessing danych przed włożeniem do kostki. Ogólnie chodzi o to, że zakładamy, że dane przyjdą w kolejności
	//ciąg za ciągiem, a zmieniamy je na kolejność pierwsze wymiary, drugie wymiary itd.
	//Dodatkowo kopiujemy dane do pamięci GPU (UWAGA: Trzeba zwolnić wektory ...Packed po dodaniu)
	thrust::device_ptr<int> DimensionsPacked;
	thrust::device_ptr<int> MeasuresPacked;
	PrepareDataForInsert(vectorsCount, dimensionsCount, dims, measuresCount, meas, &DimensionsPacked, &MeasuresPacked);

	//Wstawienie danych do kostki
	//Parametry:
	//cube - kostka
	//vectorsCount - liczba wektorów w paczce do dodania
	//DimensionsPacked - skpakowany wektor wartości wymiarów
	//MeasuresPacked - spakowany wektor wartości miar
	AddPack(cube, vectorsCount, DimensionsPacked, MeasuresPacked);

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//Zwalniamy niepotrzebne już wektory
	thrust::device_free(DimensionsPacked);
	thrust::device_free(MeasuresPacked);

	FreePack(vectorsCount, &dims, &meas);

	//Informacja o czasie wstawienia paczki danych do kostki
	printf("Data succesfully inserted. %d entries added\n", vectorsCount);
	cudaEventElapsedTime(&time, start, stop);
	printf ("Time of insert: %f ms\n\n", time);

	PrintCubeInfo(cube);
	//PrintCubeMemory(cube);	//Można wypisać do celów testowych pamięć kostki

	//Przykładowe Query do kostki - trzeba zbudować strukturę Query na podstawie reprezentacji pośredniej
	printf("Sample Querry\n");

	int operationsCount = 3;

	//Inicjalizacja struktury Query
	//Parametry:
	//dimensionsCount - liczba wszystkich wymiarów w wektorze
	//measuresCount - liczba wszystkich miar w wektorze
	//operationsCount - liczba operacji jakie chcemy wykonać na miarach w query (np. select d0 d3 sum(m0) sum(m1) max(m1) from ...
	//to są 3 operacje na miarach m0 i m1
	Querry q = InitQuerry(dimensionsCount, measuresCount, operationsCount);

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
	PrintQuerryResult(result);

	//Teraz ten wynik trzeba przesłać na serwer zapytań (najpierw trzeba przekopiować do pamięci CPU)

	//Można zwolnić query i result
	FreeResult(result);
	FreeQuerry(q);


	//Przy zabijaniu node trzeba zwolnić kostkę
	FreeCube(cube);
}

void RunIntegrationSample()
{
	printf("Integration sample ...\n\n");
	
	IR::CubeDef def {};

	IR::Dim dim1;
	dim1.range = 10000;
	def.dims.push_back(dim1);

	IR::Dim dim2;
	dim2.range = 100;
	def.dims.push_back(dim2);

	IR::Dim dim3;
	dim3.range = 500;
	def.dims.push_back(dim3);

	IR::Dim dim4;
	dim4.range = 10;
	def.dims.push_back(dim4);

	IR::Dim dim5;
	dim5.range = 500;
	def.dims.push_back(dim5);

	IR::Mea mea1;
	mea1.type = IR::Mea::Float;
	def.meas.push_back(mea1);

	IR::Mea mea2;
	mea2.type = IR::Mea::Float;
	def.meas.push_back(mea2);

	IR::Mea mea3;
	mea1.type = IR::Mea::Float;
	def.meas.push_back(mea3);

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

	IR::Query q(5, 3, 3);

	q.selectDims[0] = 1;
	q.selectDims[3] = 1;

	//Tutaj pokazujemy jakie mają być limity w where
	q.whereDimMode[0] = RTCUBE_WHERE_RANGE;	//d0 będzie brane z zakresu
	q.whereDimMode[1] = RTCUBE_WHERE_SET;		//d1 będzie brane ze zbioru
	q.whereDimMode[2] = RTCUBE_WHERE_SET;		//d2 będzie brane ze zbioru
	q.whereDimMode[3] = RTCUBE_WHERE_MAXRANGE;	//d3 jest selectowane więc jeśli nie ma dla niego nic w where,
													//to trzeba ustawić _MAXRANGE

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
	q.whereDimVals = std::vector<int>(6);
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

	//Tutaj wpisujemy dla jakich miar chcemy wykonywać operacje
	q.operationsMeasures[0] = 0;
	q.operationsMeasures[1] = 2;
	q.operationsMeasures[2] = 2;

	//Tutaj jakie to mają być operacje
	q.operationsTypes[0] = RTCUBE_OP_SUM;
	q.operationsTypes[1] = RTCUBE_OP_CNT;
	q.operationsTypes[2] = RTCUBE_OP_SUM;


	CudaCore::RTCube cube(def);
	cube.insert(rows);
	IR::QueryResult result = cube.query(q);

}

