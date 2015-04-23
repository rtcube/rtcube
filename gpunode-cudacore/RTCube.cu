#include "RTCube.cuh"

RTCube InitCube(int cubeId, int cubeCount, int dimCount, int *dimRanges, int measCount, int blocks, int threads)
{
	RTCube cube;
	cube.Id = cubeId;

	dimRanges[0] = dimRanges[0] / cubeCount;
	cube.DimCount = dimCount;
	cube.d_DimRanges = thrust::device_malloc<int>(cube.DimCount);
	thrust::copy(dimRanges, dimRanges + cube.DimCount, cube.d_DimRanges);

	cube.d_DimSizes = thrust::device_malloc<int>(cube.DimCount);
	thrust::device_vector<int> tmpDimSizes(cube.d_DimRanges, cube.d_DimRanges + dimCount);
	thrust::reverse(tmpDimSizes.begin(), tmpDimSizes.end());
	thrust::exclusive_scan(tmpDimSizes.begin(), tmpDimSizes.end(), tmpDimSizes.begin(), 1, thrust::multiplies<int>());
	thrust::reverse(tmpDimSizes.begin(), tmpDimSizes.end());
	thrust::copy(tmpDimSizes.begin(), tmpDimSizes.end(), cube.d_DimSizes);

	cube.MeasCount = measCount;
	cube.SingleMeasSize = 1;
	for(int i = 0; i < dimCount; ++i)
		cube.SingleMeasSize *= dimRanges[i];
	cube.MemorySize = cube.MeasCount * cube.SingleMeasSize;

	cube.d_CubeMemory = thrust::device_malloc<float>(cube.MemorySize);
	thrust::uninitialized_fill(thrust::device, cube.d_CubeMemory, cube.d_CubeMemory + cube.MemorySize, -1);

	cube.Blocks = blocks;
	cube.Threads = threads;

	return cube;
}

void FreeCube(RTCube cube)
{
}

__device__ int GetVecIndex(int vecNumber, int dimCount, int *dimSizes, int measCount, int vecCount, int *dims)
{
	int index = 0;

	for(int i = 0; i < dimCount; ++i)
		index += dimSizes[i] * dims[i * vecCount + vecNumber];

	return index;
}

__global__ void AddPackKernel(float *cubeMem, int dimCount, int *dimSizes, int measCount, int singleMeasSize, int vecCount, int *dims, float *meas)
{
	int currentVec = blockIdx.x * blockDim.x + threadIdx.x;

	while(currentVec < vecCount)
	{
		int currentVecIndex = GetVecIndex(currentVec, dimCount, dimSizes, measCount, vecCount, dims);

		for(int m = 0; m < measCount; ++m)
			cubeMem[m * singleMeasSize + currentVecIndex] = meas[m * vecCount + currentVec];

		currentVec += blockDim.x * gridDim.x;
	}

}

void AddPack(RTCube cube, int vecCount, thrust::device_ptr<int> d_dims, thrust::device_ptr<float> d_meas)
{
	int *dimsPtr = d_dims.get();
	float *measPtr = d_meas.get();

	float *cubeMemPtr = cube.d_CubeMemory.get();
	int *dimSizesPtr = cube.d_DimSizes.get();

	AddPackKernel <<<cube.Blocks, cube.Threads>>>(cubeMemPtr, cube.DimCount, dimSizesPtr, cube.MeasCount, cube.SingleMeasSize, vecCount, dimsPtr, measPtr);
	cudaDeviceSynchronize();
	gpuErrChk(cudaPeekAtLastError());
}

void PrintCubeInfo(RTCube cube)
{
	printf("ID: %3d\n", cube.Id);
	printf("Memory size: %10d bytes\n", cube.MemorySize * sizeof(float));
	printf("Number of dimensions: %3d\n", cube.DimCount);
	printf("Number of measures: %3d\n", cube.MeasCount);
	printf("Single measure size: %3d\n", cube.SingleMeasSize);
	printf("\n");
}

void PrintCubeMemory(RTCube cube)
{
	for(int m = 0; m < cube.MeasCount; ++m)
	{
		for(int i = 0; i < cube.SingleMeasSize; ++i)
			printf("%15f", (float)cube.d_CubeMemory[m * cube.SingleMeasSize + i]);
		printf("\n");
	}
	printf("\n");
}

Querry InitQuerry(int dimCount, int measCount)
{
	Querry querry;
	querry.DimCount = dimCount;
	querry.MeasCount = measCount;

	querry.d_SelectDims = thrust::device_malloc<int>(dimCount);
	thrust::fill_n(querry.d_SelectDims, dimCount, 0);

	querry.d_SelectMeasOps = thrust::device_malloc<int>(measCount);
	thrust::fill_n(querry.d_SelectMeasOps, measCount, 0);

	querry.d_WhereDimValsCounts = thrust::device_malloc<int>(dimCount);
	thrust::fill_n(querry.d_WhereDimValsCounts, dimCount, 0);

	querry.d_WhereDimValsStart = thrust::device_malloc<int>(dimCount);
	thrust::fill_n(querry.d_WhereDimValsStart, dimCount, 0);

	return querry;
}

QueryResult ProcessQuerry(RTCube cube, Querry querry)
{
	QueryResult result;


	return result;
}