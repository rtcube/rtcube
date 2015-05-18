#include "RTCube.cuh"

RTCube InitCube(int cubeId, int cubeCount, int dimCount, int *dimRanges, int measCount, int blocks, int threads)
{
	RTCube cube;
	cube.Id = cubeId;

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

	cube.d_CubeMemory = thrust::device_malloc<int>(cube.MemorySize);
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

__global__ void AddPackKernel(int *cubeMem, int dimCount, int *dimSizes, int measCount, int singleMeasSize, int vecCount, int *dims, int *meas)
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

void AddPack(RTCube cube, int vecCount, thrust::device_ptr<int> d_dims, thrust::device_ptr<int> d_meas)
{
	int *dimsPtr = d_dims.get();
	int *measPtr = d_meas.get();

	int *cubeMemPtr = cube.d_CubeMemory.get();
	int *dimSizesPtr = cube.d_DimSizes.get();

	AddPackKernel <<<cube.Blocks, cube.Threads>>>(cubeMemPtr, cube.DimCount, dimSizesPtr, cube.MeasCount, cube.SingleMeasSize, vecCount, dimsPtr, measPtr);
	cudaDeviceSynchronize();
	gpuErrChk(cudaPeekAtLastError());
}

void PrintCubeInfo(RTCube cube)
{
	printf("ID: %3d\n", cube.Id);
	printf("Memory size: %10d bytes\n", cube.MemorySize * sizeof(int));
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
			printf("%4d", (int)cube.d_CubeMemory[m * cube.SingleMeasSize + i]);
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

	querry.d_WhereDimValuesCounts = thrust::device_malloc<int>(dimCount);
	thrust::fill_n(querry.d_WhereDimValuesCounts, dimCount, 0);

	querry.d_WhereDimValsStart = thrust::device_malloc<int>(dimCount);
	thrust::fill_n(querry.d_WhereDimValsStart, dimCount, 0);

	querry.d_WhereDimMode = thrust::device_malloc<int>(dimCount);
	thrust::fill_n(querry.d_WhereDimMode, dimCount, RTCUBE_WHERE_NONE);

	querry.d_WhereStartRange = thrust::device_malloc<int>(dimCount);
	thrust::fill_n(querry.d_WhereStartRange, dimCount, 0);

	querry.d_WhereEndRange = thrust::device_malloc<int>(dimCount);
	thrust::fill_n(querry.d_WhereEndRange, dimCount, 0);

	return querry;
}

void PrintQuerry(Querry q)
{
	printf("SELECT ");
	for (int i = 0; i < q.DimCount; ++i)
		if ((int)q.d_SelectDims[i] != 0)
			printf("d%d ", i);

	for (int i = 0; i < q.MeasCount; ++i)
		if ((int)q.d_SelectMeasOps[i] != RTCUBE_OP_NONE)
		{
			switch ((int)q.d_SelectMeasOps[i])
			{
			case RTCUBE_OP_SUM:
				printf("SUM(m%d) ", i);
				break;
			case RTCUBE_OP_MAX:
				printf("MAX(m%d) ", i);
				break;
			case RTCUBE_OP_MIN:
				printf("MIN(m%d) ", i);
				break;
			case RTCUBE_OP_AVG:
				printf("AVG(m%d) ", i);
				break;
			case RTCUBE_OP_CNT:
				printf("CNT(m%d) ", i);
				break;
			default:
				printf("ERROR - UNKNOWN OP\n");
				return;
			}
		}

	printf("FROM CUBE\n");

	int wherePrinted = 0;
	for (int i = 0; i < q.DimCount; ++i)
	{
		int mode = (int)q.d_WhereDimMode[i];
		if (mode != RTCUBE_WHERE_NONE)
		{
			if (mode == RTCUBE_WHERE_SET)
			{
				if (wherePrinted == 0)
				{
					printf("WHERE ");
					wherePrinted = 1;
				}
				else
					printf("AND ");

				printf("d%d IN { ", i);
				for (int j = 0; j < (int)q.d_WhereDimValuesCounts[i]; ++j)
					printf("%d ", (int)q.d_WhereDimVals[(int)q.d_WhereDimValsStart[i] + j]);

				printf("}\n");
			}
			else if (mode == RTCUBE_WHERE_RANGE)
			{
				if (wherePrinted == 0)
				{
					printf("WHERE ");
					wherePrinted = 1;
				}
				else
					printf("AND ");

				printf("d%d IN <%d %d>\n", i, (int)q.d_WhereStartRange[i], (int)q.d_WhereEndRange[i]);
			}

		}
		
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

	int resultSize = 0;
	for (int i = 0; i < q.MeasCount; ++i)
		if ((int)q.d_SelectMeasOps[i] != RTCUBE_OP_NONE)
			++resultSize;

	result.MeasPerResult = resultSize;

	result.d_ResultMeas = thrust::device_malloc<int>(result.ResultsCount * result.MeasPerResult);


	return result;
}

__device__ bool CheckDimValueAgainsQuerry(int dim, int dimValue, Querry querry)
{
	if (querry.d_WhereDimMode[dim] == RTCUBE_WHERE_RANGE || querry.d_WhereDimMode[dim] == RTCUBE_WHERE_MAXRANGE)
		return dimValue >= querry.d_WhereStartRange[dim] && dimValue <= querry.d_WhereEndRange[dim];
	else if (querry.d_WhereDimMode[dim] == RTCUBE_WHERE_SET)
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
	int cellIndex = blockIdx.x * blockDim.x + threadIdx.x;

	while (cellIndex < cube.SingleMeasSize)
	{
		int currentCellIndex = cellIndex;
		int resultIndex = 0;

		//Przegl¹damy aktualn¹ komórkê kostki po wszystkich wymiarach, wyliczamy jej wartoœci na kolejnych wymiarach i patrzymy,
		//czy spe³niaj¹ warunki zapytania. Równoczeœnie budujemy indeks w rozwi¹zaniu na podstawie wartoœci wymiarów.

		int dim;
		int currentDimValue;
		for (dim = 0; dim < cube.DimCount; ++dim)
		{
			currentDimValue = currentCellIndex / cube.d_DimRanges[dim];

			//Sprawdzamy, czy wartoœæ wymiaru wchodzi w zakres querry
			if (CheckDimValueAgainsQuerry(dim, currentDimValue, querry))
			{
				int a = currentDimValue;
				//Jeœli aktualny wymiar jest selectowany, aktualizujemy wartoœæ indeksu w rozwi¹zaniu
				//if (querry.d_SelectDims[dim] == 1)
				//{
				//	//Szukamy indeksu w result aktualnej wartoœci 
				//	int currentDimValueIndex;
				//	if (querry.d_WhereDimMode[dim] == RTCUBE_WHERE_MAXRANGE || querry.d_WhereDimMode[dim] == RTCUBE_WHERE_RANGE)
				//		currentDimValueIndex = currentDimValue - querry.d_WhereDimValsStart[dim];
				//	else if (querry.d_WhereDimMode[dim] == RTCUBE_WHERE_SET)
				//	{
				//		currentDimValueIndex = 0;
				//		while (querry.d_WhereDimVals[querry.d_WhereDimValsStart[dim] + currentDimValueIndex] != currentDimValue)
				//			++currentDimValueIndex;
				//	}

				//	resultIndex += result.d_SelectDimSizes[dim] * currentDimValueIndex;

				//}
			}
			else
				break;
			
			currentCellIndex %= cube.d_DimRanges[dim];
		}

		//Sprawdzamy, czy aktualna komórka wchodzi to rozwi¹zania
		if (dim >= cube.DimCount)
		{
			//Przechodzimy po wszystkich miarach i aktualizujemy wynik odpowiedni¹ operacj¹

			//result.d_ResultMeas[0 * result.ResultsCount + resultIndex] = 1;
			result.d_ResultMeas[0] = result.d_ResultMeas[0] + 1;;

		}


		cellIndex += gridDim.x * blockDim.x;
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

void PrintQuerryResult(QueryResult result)
{
	printf("Querry result\n");
	printf("Result lines count:%4d\n", result.ResultsCount);
	printf("Meas per result:%4d\n\n", result.MeasPerResult);

	for (int i = 0; i < result.ResultsCount; ++i)
	{
		int index = i;

		printf("%3d: ", i);

		for (int j = 0; j < result.Q.DimCount; ++j)
		{
			if ((int)result.Q.d_SelectDims[j] == 1)
			{
				int dimValIndex =  index / result.d_SelectDimSizes[j];
				int dimVal;
				if ((int)result.Q.d_WhereDimMode[j] == RTCUBE_WHERE_SET)
					dimVal = (int)result.Q.d_WhereDimVals[(int)result.Q.d_WhereDimValsStart[j] + dimValIndex];
				else
					dimVal = (int)result.Q.d_WhereStartRange[j] + dimValIndex;

				printf("%3d", dimVal);

				index %= result.d_SelectDimSizes[j];
			}
		}

		printf("  :");

		for (int j = 0; j < result.MeasPerResult; ++j)
			printf("%3d", (int)result.d_ResultMeas[j * result.ResultsCount + i]);

		printf("\n");
	}
}