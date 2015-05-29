#include "RTCube.cuh"

RTCube InitCube(float cardMemoryPartToFill, int dimensionsCount, int *dimensionsSizes, int measuresCount, int blocks, int threads)
{

	RTCube cube;
	cube.DimensionsCount = dimensionsCount;
	cube.MeasuresCount = measuresCount;

	cube.DimensionsRanges = thrust::device_malloc<int>(dimensionsCount);
	thrust::copy_n(dimensionsSizes, dimensionsCount, cube.DimensionsRanges);

	cube.DimensionsSizes = thrust::device_malloc<int>(cube.DimensionsCount);
	thrust::device_vector<int> tmpDimSizes(cube.DimensionsRanges, cube.DimensionsRanges + dimensionsCount);
	thrust::reverse(tmpDimSizes.begin(), tmpDimSizes.end());
	thrust::exclusive_scan(tmpDimSizes.begin(), tmpDimSizes.end(), tmpDimSizes.begin(), 1, thrust::multiplies<int>());
	thrust::reverse(tmpDimSizes.begin(), tmpDimSizes.end());
	thrust::copy(tmpDimSizes.begin(), tmpDimSizes.end(), cube.DimensionsSizes);

	cube.MemoryPerVector = sizeof(unsigned long int) + measuresCount * sizeof(int);
	size_t freeMemory, totalMemory;
	cudaMemGetInfo(&freeMemory, &totalMemory);

	double memoryToAllocate = (double)freeMemory * (double)cardMemoryPartToFill;
	cube.Capacity = (int)memoryToAllocate / cube.MemoryPerVector;
	cube.VectorsCount = 0;

	cube.Codes = thrust::device_malloc<unsigned long int>(cube.Capacity);
	cube.Measures = thrust::device_malloc<int>(cube.Capacity * measuresCount);

	cube.Blocks = blocks;
	cube.Threads = threads;

	return cube;
}

__device__ int GetVecIndex(int vecNumber, int dimCount, int *dimSizes, int measCount, int vecCount, int *dims)
{
	unsigned long int index = 0;

	for (int i = 0; i < dimCount; ++i)
		index += (unsigned long int)dimSizes[i] * (unsigned long int)dims[i * vecCount + vecNumber];

	return index;
}

__global__ void AddPackKernel(unsigned long int *codes, int *measures, int dimensionsCount, int *dimendionsSizes, int measuresCount, int currentCapacity, int fullCapacity,
	int packCount, int *packDimensions, int *packMeasures)
{
	int currentVec = blockIdx.x * blockDim.x + threadIdx.x;

	while (currentVec < packCount)
	{
		codes[currentCapacity + currentVec] = GetVecIndex(currentVec, dimensionsCount, dimendionsSizes, measuresCount, packCount, packDimensions);

		for (int i = 0; i < measuresCount; ++i)
			measures[i * fullCapacity + currentCapacity + currentVec] = packMeasures[i * packCount + currentVec];

		currentVec += blockDim.x * gridDim.x;
	}

}

void AddPack(RTCube &cube, int vecCount, thrust::device_ptr<int> d_dims, thrust::device_ptr<int> d_meas)
{
	int *dimsPtr = d_dims.get();
	int *measPtr = d_meas.get();

	unsigned long int *codesPtr = cube.Codes.get();
	int *measuresPtr = cube.Measures.get();
	int *dimensionsSizesPtr = cube.DimensionsSizes.get();

	AddPackKernel << < cube.Blocks, cube.Threads >> > (codesPtr, measuresPtr, cube.DimensionsCount, dimensionsSizesPtr, cube.MeasuresCount,
		cube.VectorsCount, cube.Capacity, vecCount, dimsPtr, measPtr);

	cudaDeviceSynchronize();
	gpuErrChk(cudaPeekAtLastError());

	cube.VectorsCount += vecCount;
}

void FreeCube(RTCube cube)
{
	thrust::device_free(cube.DimensionsRanges);
	thrust::device_free(cube.Codes);
	thrust::device_free(cube.Measures);
}

void PrintCubeInfo(RTCube cube)
{
	printf("Number of dimensions: %d\n", cube.DimensionsCount);
	printf("Dimensions ranges: ");
	for (int i = 0; i < cube.DimensionsCount; ++i)
		printf(" < 0 ,  %d ) ;", (int)cube.DimensionsRanges[i]);
	printf("\n");
	for (int i = 0; i < cube.DimensionsCount; ++i)
		printf("%d; ", (int)cube.DimensionsSizes[i]);
	printf("\n");
	printf("Measures count: %d\n", cube.MeasuresCount);
	printf("Memory per vector: %d\n", cube.MemoryPerVector);
	printf("Cube capacity: %d\n", cube.Capacity);
	printf("Currently inserted vectors count: %d\n\n", cube.VectorsCount);
}

void PrintCubeMemory(RTCube cube)
{
	printf("Cube memory:\n");
	for (int i = 0; i < cube.VectorsCount; ++i)
	{
		printf("%10lu ", (unsigned long int)cube.Codes[i]);
		for (int m = 0; m < cube.MeasuresCount; ++m)
			printf("%4d", (int)cube.Measures[m * cube.Capacity + i]);
		printf("\n");
	}


	printf("\n");
}

QueryResult InitResult(RTCube cube, Querry q)
{
	QueryResult result;

	result.Q = q;

	result.d_SelectDimSizes = thrust::device_malloc<int>(q.DimCount);

	int currentSize = 1;
	for (int i = q.DimCount - 1; i >= 0; --i)
	{
		if (q.d_SelectDims[i] != 0)
		{
			result.d_SelectDimSizes[i] = currentSize;
			currentSize *= q.d_WhereDimValuesCounts[i];
		}
	}

	result.ResultsCount = currentSize;

	result.MeasPerResult = q.OperationsCount;

	result.d_ResultMeas = thrust::device_malloc<int>(result.ResultsCount * result.MeasPerResult);
	thrust::fill_n(result.d_ResultMeas, result.ResultsCount * result.MeasPerResult, 0);


	return result;
}

__device__ bool CheckDimValueAgainsQuerry(int dim, int dimValue, Querry querry)
{
	if (querry.d_WhereDimMode[dim] == RTCUBE_WHERE_RANGE || querry.d_WhereDimMode[dim] == RTCUBE_WHERE_MAXRANGE)
		return dimValue >= querry.d_WhereStartRange[dim] && dimValue <= querry.d_WhereEndRange[dim];
	if (querry.d_WhereDimMode[dim] == RTCUBE_WHERE_SET)
	{
		bool result = false;
		for (int i = 0; i < querry.d_WhereDimValuesCounts[dim]; ++i)
			if (dimValue == querry.d_WhereDimVals[querry.d_WhereDimValsStart[dim] + i])
			{
				result = true;
				break;
			}

		return result;
	}

	return true;
}

__global__ void ProcessQuerryKernel(RTCube cube, Querry querry, QueryResult result)
{
	int currentVector = blockIdx.x * blockDim.x + threadIdx.x;

	while (currentVector < cube.VectorsCount)
	{
		unsigned long int currentCode = cube.Codes[currentVector];

		int indexInResult = 0;

		for (int i = 0; i < cube.DimensionsCount; ++i)
		{
			int dimValue = (int)(currentCode / cube.DimensionsSizes[i]);

			if (!CheckDimValueAgainsQuerry(i, dimValue, querry))
			{
				indexInResult = -1;
				break;
			}

			if (querry.d_SelectDims[i] == 1)
				indexInResult += result.d_SelectDimSizes[i] * dimValue;


			currentCode %= (unsigned long int)cube.DimensionsSizes[i];

		}

		if (indexInResult != -1)
		{
			for (int i = 0; i < querry.OperationsCount; ++i)
			{
				int m = querry.OperationsMeasures[i];
				int operation = querry.OperationsTypes[i];

				int *resultAddres = result.d_ResultMeas.get() + (i * result.ResultsCount + indexInResult);

				switch (operation)
				{
				case RTCUBE_OP_CNT:
					atomicAdd(resultAddres, 1);
					break;

				case RTCUBE_OP_MAX:
					atomicMax(resultAddres, cube.Measures[m * cube.Capacity + currentVector]);
					break;

				case RTCUBE_OP_MIN:
					atomicMin(resultAddres, cube.Measures[m * cube.Capacity + currentVector]);
					break;

				case RTCUBE_OP_SUM:
					atomicAdd(resultAddres, cube.Measures[m * cube.Capacity + currentVector]);

					break;
				}
			}

			
		}

		currentVector += gridDim.x * blockDim.x;
	}
}

QueryResult ProcessQuerry(RTCube cube, Querry querry)
{
	QueryResult result = InitResult(cube, querry);

	ProcessQuerryKernel << <cube.Blocks, cube.Threads >> >(cube, querry, result);
	cudaDeviceSynchronize();
	gpuErrChk(cudaPeekAtLastError());

	return result;
}
