#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "RTCube.cuh"

void PrintVector(int dimCount, int *dimVals, int measCount, int *measVals)
{
	for(int i = 0; i < dimCount; ++i)
		printf("%4d", dimVals[i]);
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

void PrintPackedPack(int vecCount, int dimCount, int *dims, int measCount, int *meas)
{
	for(int i = 0; i < vecCount; ++i)
	{
		for(int j = 0; j < dimCount; ++j)
			printf("%4d", dims[j * vecCount + i]);
		for(int j = 0; j < measCount; ++j)
			printf("%4d", meas[j * vecCount + i]);
		printf("\n");
	}
	printf("\n");
}

void GeneretVector(int cubeId, int cubeCount, int dimCount, int *dimRanges, int measCount, int measMax, int **dimVals, int **measVals)
{
	*dimVals = (int*)malloc(dimCount * sizeof(int));
	*measVals = (int*)malloc(measCount * sizeof(int));

	for(int i = 0; i < dimCount; ++i)
		(*dimVals)[i] = rand() % dimRanges[i];

	for(int i = 0; i < measCount; ++i)
		(*measVals)[i] = rand() % measMax;
}

void GeneratePack(int cubeId, int cubeCount, int vecCount, int dimCount, int *dimRanges, int measCount, int measMax, int ***dims, int ***meas)
{
	*dims = (int**)malloc(vecCount * sizeof(int*));
	*meas = (int**)malloc(vecCount * sizeof(int*));

	for(int i = 0; i < vecCount; ++i)
		GeneretVector(cubeId, cubeCount, dimCount, dimRanges, measCount, measMax, &((*dims)[i]), &((*meas)[i]));
}

void PrepareDataForInsert(int cubeId, int cubeCount, int vecCount, int dimCount, int **dims, int measCount, int **meas, thrust::device_ptr<int> *d_dimsPacked, thrust::device_ptr<int> *d_measPacked)
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

	//TODO:Ewentualne usuwanie duplikatów

	free(h_dimsPacked);
	free(h_measPacked);
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

int main()
{
	srand(time(NULL));

	int cubeId = 3;
	int cubeCount = 10;

	//Wygenerowanie przykładowych danych
	int vecCount = 20000;
	int dimCount = 6;
	int measCount = 3;

	int *dimRanges = (int*)malloc(dimCount * sizeof(int));
	dimRanges[0] = 20;
	dimRanges[1] = 5;
	dimRanges[2] = 6;
	dimRanges[3] = 3;
	dimRanges[4] = 5;
	dimRanges[5] = 4;

	//dimCount = 3;
	//dimRanges[0] = 10;
	//dimRanges[1] = 2;
	//dimRanges[2] = 3;

	int measMax = 100;

	int **dims;
	int **meas;

	GeneratePack(cubeId, cubeCount, vecCount, dimCount, dimRanges, measCount, measMax, &dims, &meas);
	//PrintPack(vecCount, dimCount, measCount, dims, meas);

	//Inicjalizacja kostki na GPU
	RTCube cube = InitCube(cubeId, cubeCount, dimCount, dimRanges, measCount, 64, 64);
	printf("Cube succesfully inititialized\n");
	PrintCubeInfo(cube);

	//Preprocessing danych przed włożeniem do kostki
	thrust::device_ptr<int> d_dimsPacked;
	thrust::device_ptr<int> d_measPacked;
	PrepareDataForInsert(cubeId, cubeCount, vecCount, dimCount, dims, measCount, meas, &d_dimsPacked, &d_measPacked);

	//Umieszczenie danych w kostce
	AddPack(cube, vecCount, d_dimsPacked, d_measPacked);
	printf("Data succesfully inserted\n\n");

	//PrintCubeMemory(cube);

	//Przykładowe kwerendy
	printf("Sample Querry\n");

	//select d0 d3 SUM(m0) MAX(m2) from cube
	//where d0 in {0, 1, 3}
	//and d1 in {0, 2, 4}
	//and d2 in <1, 2>

	Querry q = InitQuerry(dimCount, measCount);
	q.d_SelectDims[0] = 1;
	q.d_SelectDims[3] = 1;

	//q.d_SelectDims[1] = 1;

	q.d_SelectMeasOps[0] = RTCUBE_OP_SUM;
	q.d_SelectMeasOps[2] = RTCUBE_OP_MAX;

	q.d_WhereDimMode[0] = RTCUBE_WHERE_RANGE;
	q.d_WhereDimMode[1] = RTCUBE_WHERE_SET;
	q.d_WhereDimMode[2] = RTCUBE_WHERE_SET;
	q.d_WhereDimMode[3] = RTCUBE_WHERE_MAXRANGE;

	q.d_WhereDimValuesCounts[0] = 20;
	q.d_WhereDimValuesCounts[1] = 3;
	q.d_WhereDimValuesCounts[2] = 3;
	q.d_WhereDimValuesCounts[3] = dimRanges[3];

	q.d_WhereStartRange[0] = 0;
	q.d_WhereStartRange[3] = 0;

	q.d_WhereEndRange[0] = 19;
	q.d_WhereEndRange[3] = dimRanges[3];

	q.d_WhereDimValsStart = thrust::device_malloc<int>(dimCount);
	q.d_WhereDimValsStart[1] = 0;
	q.d_WhereDimValsStart[2] = 3;

	q.d_WhereDimVals = thrust::device_malloc<int>(6);
	q.d_WhereDimVals[0] = 0;
	q.d_WhereDimVals[1] = 1;
	q.d_WhereDimVals[2] = 3;
	q.d_WhereDimVals[3] = 0;
	q.d_WhereDimVals[4] = 2;
	q.d_WhereDimVals[5] = 4;

	PrintQuerry(q);

	QueryResult result = ProcessQuerry(cube, q);

	PrintQuerryResult(result);

	FreePack(vecCount, &dims, &meas);
	FreeCube(cube);

	return 0;
}



