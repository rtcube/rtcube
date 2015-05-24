#include "RTCube.cuh"
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

void cubeQuery()
{

}

void cubeStatus()
{
	PrintCubeInfo(cube);
}
