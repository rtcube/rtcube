#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/random.h>
#include <thrust/advance.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

typedef struct RTCube
{
	int Id;
	int MemorySize;
	thrust::device_ptr<float> d_CubeMemory;

	int DimCount;
	thrust::device_ptr<int> d_DimRanges;
	thrust::device_ptr<int> d_DimSizes;

	int MeasCount;
	int SingleMeasSize;

	int Blocks;
	int Threads;

}RTCube;

#define RTCUBE_OP_SUM 1
#define RTCUBE_OP_MAX 2
#define RTCUBE_OP_MIN 3
#define RTCUBE_OP_AVG 4

typedef struct Querry
{
	thrust::device_ptr<int> d_SelectDims;

	int DimCount;

	thrust::device_ptr<int> d_SelectMeasOps;

	thrust::device_ptr<int> d_WhereDimValsCounts;

	thrust::device_ptr<int> d_WhereDimVals;

	thrust::device_ptr<int> d_WhereDimValsStart;

	int MeasCount;

}Querry;

typedef struct QueryResult
{

}QueryResult;

RTCube InitCube(int cubeId, int cubeCount, int dimCount, int *dimRanges, int measCount, int blocks, int threads);

void FreeCube(RTCube cube);

void PrintCubeInfo(RTCube cube);

void PrintCubeMemory(RTCube cube);

void AddPack(RTCube cube, int vecCount, thrust::device_ptr<int> d_dims, thrust::device_ptr<float> d_meas);

Querry InitQuerry(int dimCount, int measCount);

QueryResult ProcessQuerry(RTCube cube, Querry querry);

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

