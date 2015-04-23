#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "RTCube.cuh"

void PrintVector(int dimCount, int *dimVals, int measCount, float *measVals)
{
	for(int i = 0; i < dimCount; ++i)
		printf("%4d", dimVals[i]);
	for(int i = 0; i < measCount; ++i)
		printf("%15f", measVals[i]);
	printf("\n");
}

void PrintPack(int vecCount, int dimCount, int measCount, int **dims, float **meas)
{
	for(int i = 0; i < vecCount; ++i)
		PrintVector(dimCount, dims[i], measCount, meas[i]);
	printf("\n");
}

void PrintPackedPack(int vecCount, int dimCount, int *dims, int measCount, float *meas)
{
	for(int i = 0; i < vecCount; ++i)
	{
		for(int j = 0; j < dimCount; ++j)
			printf("%4d", dims[j * vecCount + i]);
		for(int j = 0; j < measCount; ++j)
			printf("%15f", meas[j * vecCount + i]);
		printf("\n");
	}
	printf("\n");
}

void GeneretVector(int cubeId, int cubeCount, int dimCount, int *dimRanges, int measCount, float measMax, int **dimVals, float **measVals)
{
	*dimVals = (int*)malloc(dimCount * sizeof(int));
	*measVals = (float*)malloc(measCount * sizeof(float));

	(*dimVals)[0] = (rand() % (dimRanges[0] / cubeCount)) * cubeCount + cubeId;

	for(int i = 1; i < dimCount; ++i)
		(*dimVals)[i] = rand() % dimRanges[i];

	for(int i = 0; i < measCount; ++i)
		(*measVals)[i] = (float)rand()/(float)(RAND_MAX/measMax);
}

void GeneratePack(int cubeId, int cubeCount, int vecCount, int dimCount, int *dimRanges, int measCount, float measMax, int ***dims, float ***meas)
{
	*dims = (int**)malloc(vecCount * sizeof(int*));
	*meas = (float**)malloc(vecCount * sizeof(float*));

	for(int i = 0; i < vecCount; ++i)
		GeneretVector(cubeId, cubeCount, dimCount, dimRanges, measCount, measMax, &((*dims)[i]), &((*meas)[i]));
}

void PrepareDataForInsert(int cubeId, int cubeCount, int vecCount, int dimCount, int **dims, int measCount, float **meas, thrust::device_ptr<int> *d_dimsPacked, thrust::device_ptr<float> *d_measPacked)
{
	int *h_dimsPacked = (int*)malloc(vecCount * dimCount * sizeof(int));
	float *h_measPacked = (float*)malloc(vecCount * measCount * sizeof(float));

	for(int i = 0; i < vecCount; ++i)
	{
		h_dimsPacked[i] = (dims[i][0] - cubeId) / cubeCount;

		for(int j = 1; j < dimCount; ++j)
			h_dimsPacked[j * vecCount + i] = dims[i][j];

		for(int j = 0; j < measCount; ++j)
			h_measPacked[j * vecCount + i] = meas[i][j];
	}

	//PrintPackedPack(vecCount, dimCount, h_dimsPacked, measCount, h_measPacked);

	*d_dimsPacked = thrust::device_malloc<int>(vecCount * dimCount);
	thrust::copy(h_dimsPacked, h_dimsPacked + vecCount * dimCount, *d_dimsPacked);

	*d_measPacked = thrust::device_malloc<float>(vecCount * measCount);
	thrust::copy(h_measPacked, h_measPacked + vecCount * measCount, *d_measPacked);

	//TODO:Ewentualne usuwanie duplikatów

	free(h_dimsPacked);
	free(h_measPacked);
}

void FreePack(int vecCount, int ***dims, float ***meas)
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
	int vecCount = 2000;
	int dimCount = 4;
	int measCount = 3;

	int *dimRanges = (int*)malloc(4 * sizeof(int));
	dimRanges[0] = 200;
	dimRanges[1] = 5;
	dimRanges[2] = 6;
	dimRanges[3] = 3;
	float measMax = 100.0;

	int **dims;
	float **meas;

	GeneratePack(cubeId, cubeCount, vecCount, dimCount, dimRanges, measCount, measMax, &dims, &meas);
	//PrintPack(vecCount, dimCount, measCount, dims, meas);

	//Inicjalizacja kostki na GPU
	RTCube cube = InitCube(cubeId, cubeCount, dimCount, dimRanges, measCount, 64, 64);
	printf("Cube succesfully inititialized\n");
	PrintCubeInfo(cube);

	//Preprocessing danych przed włożeniem do kostki
	thrust::device_ptr<int> d_dimsPacked;
	thrust::device_ptr<float> d_measPacked;
	PrepareDataForInsert(cubeId, cubeCount, vecCount, dimCount, dims, measCount, meas, &d_dimsPacked, &d_measPacked);

	//Umieszczenie danych w kostce
	AddPack(cube, vecCount, d_dimsPacked, d_measPacked);
	printf("Data succesfully inserted\n");

	//PrintCubeMemory(cube);

	//Przykładowe kwerendy

	//select d0 SUM(m0) MAX(m2) from cube
	//where d0 in {0, 1, 3}
	//and d2 in {0, 2}
	//and d3 in {0, 1}

	Querry q = InitQuerry(dimCount, measCount);
	q.d_SelectDims[0] = 1;
	q.d_SelectMeasOps[0] = RTCUBE_OP_SUM;
	q.d_SelectMeasOps[2] = RTCUBE_OP_MAX;
	q.d_WhereDimValsCounts[0] = 3;
	q.d_WhereDimValsCounts[2] = 2;
	q.d_WhereDimValsCounts[3] = 2;
	q.d_WhereDimVals = thrust::device_malloc<int>(3 + 2 + 2);
	q.d_WhereDimVals[0] = 0; q.d_WhereDimVals[1] = 1; q.d_WhereDimVals[2] = 3;
	q.d_WhereDimVals[3] = 0; q.d_WhereDimVals[4] = 2;
	q.d_WhereDimVals[5] = 0; q.d_WhereDimVals[6] = 1;
	thrust::exclusive_scan(q.d_WhereDimValsCounts, q.d_WhereDimValsCounts + q.DimCount, q.d_WhereDimValsStart);

	QueryResult result = ProcessQuerry(cube, q);

	FreePack(vecCount, &dims, &meas);
	FreeCube(cube);

	return 0;
}



