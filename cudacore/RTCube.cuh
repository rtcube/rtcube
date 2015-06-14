#pragma once

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

#include "RTQuery.cuh"

#define gpuErrChk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

typedef struct RTCube
{
	//Ile wektorów przy aktualnej definicji kostka może pomieścić
	uint64_t Capacity;

	//Pamięć, gdzie trzymane są kody wektorów
	thrust::device_ptr<unsigned long int> Codes;

	//Pamięć, gdzie trzymane są wartości miar wektorów
	thrust::device_ptr<int> Measures;

	//Liczba ciągów aktulanie znajdujących się w kostce
	int VectorsCount;

	//Liczba wymiarów kostki
	int DimensionsCount;

	//Zakresy wymiarów, czyli zakresy wartości (dla i-tego wymiaru < 0 , DimensionSizes[i] )
	thrust::device_ptr<int> DimensionsRanges;

	//Rozmiary wymiarów w wielowymiaroewj tablicy - informacja potrzebna do indeksowania
	thrust::device_ptr<int> DimensionsSizes;

	//Liczba miar w wektorach
	int MeasuresCount;

	//Rozmiar w pamięci - w bajtach
	int MemoryPerVector;

	int Blocks;
	int Threads;

}RTCube;

RTCube InitCube(float cardMemoryPartToFill, int dimensionsCount, int *dimensionsSizes, int measuresCount, int blocks, int threads);

void AddPack(RTCube &cube, int vecCount, thrust::device_ptr<int> d_dims, thrust::device_ptr<int> d_meas);

void FreeCube(RTCube cube);

void PrintCubeInfo(RTCube cube);

void PrintCubeMemory(RTCube cube);

QueryResult ProcessQuerry(RTCube cube, Querry querry);

