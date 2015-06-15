#include "api.cuh"

#include "RTCube.cuh"
#include <vector>

#define MEM_TO_FILL 0.2
#define BLOCKS 32
#define THREADS 32

CudaCube::CudaCube(const IR::CubeDef& cubeDef): CubeImpl(cubeDef)
{
	int *dimensionsSizes = (int*)malloc(sizeof(int)*cubeDef.dims.size());
	for(int i = 0; i < cubeDef.dims.size(); ++i)
		dimensionsSizes[i] = cubeDef.dims[i].range;

	cube = InitCube(MEM_TO_FILL, cubeDef.dims.size(), dimensionsSizes, cubeDef.meas.size(), BLOCKS, THREADS);
	free(dimensionsSizes);

	//PrintCubeInfo(cube);
}

CudaCube::~CudaCube()
{
	FreeCube(cube);
}

void CudaCube::insert(const IR::Rows& rows)
{
	thrust::device_ptr<int> DimensionsPacked = thrust::device_malloc<int>(rows.num_dims * rows.num_rows);
	thrust::device_ptr<int> MeasuresPacked = thrust::device_malloc<int>(rows.num_meas * rows.num_rows);

	std::vector<int> hostDimenisons;
	std::vector<int> hostMeasures;

	for(int i = 0; i < rows.num_dims; ++i)
		for(int j = 0; j < rows.num_rows; ++j)
			hostDimenisons.push_back(rows.dims[j * rows.num_dims + i]);

	for(int i = 0; i < rows.num_meas; ++i)
	{
		IR::Mea::Type type = def().meas[i].type;
		
		if(type == IR::Mea::Int)
			for(int j = 0; j < rows.num_rows; ++j)
				hostMeasures.push_back(rows.meas[j * rows.num_meas + i].i);
		else
			for(int j = 0; j < rows.num_rows; ++j)
				hostMeasures.push_back(rows.meas[j * rows.num_meas + i].f);
	}
		
	thrust::copy_n(hostDimenisons.begin(), rows.num_dims * rows.num_rows, DimensionsPacked);
	thrust::copy_n(hostMeasures.begin(), rows.num_meas * rows.num_rows, MeasuresPacked);

	AddPack(cube, rows.num_rows, DimensionsPacked, MeasuresPacked);

	thrust::device_free(DimensionsPacked);
	thrust::device_free(MeasuresPacked);

	//PrintCubeInfo(cube);
	//PrintCubeMemory(cube);

}

#include <iostream>
IR::QueryResult CudaCube::query(const IR::Query& q)
{
	Querry cudaQuery = InitQuerry(def().dims.size(), def().meas.size(), q.operationsMeasures.size());

	thrust::copy(q.selectDims.begin(), q.selectDims.end(), cudaQuery.d_SelectDims);
	thrust::copy(q.operationsMeasures.begin(), q.operationsMeasures.end(), cudaQuery.OperationsMeasures);
	thrust::copy(q.operationsTypes.begin(), q.operationsTypes.end(), cudaQuery.OperationsTypes);
	thrust::copy(q.whereDimMode.begin(), q.whereDimMode.end(), cudaQuery.d_WhereDimMode);
	cudaQuery.d_WhereDimVals = thrust::device_malloc<int>(q.whereDimVals.size());
	thrust::copy(q.whereDimVals.begin(), q.whereDimVals.end(), cudaQuery.d_WhereDimVals);
	thrust::copy(q.whereDimValsStart.begin(), q.whereDimValsStart.end(), cudaQuery.d_WhereDimValsStart);
	thrust::copy(q.whereDimValuesCounts.begin(), q.whereDimValuesCounts.end(), cudaQuery.d_WhereDimValuesCounts);
	thrust::copy(q.whereEndRange.begin(), q.whereEndRange.end(), cudaQuery.d_WhereEndRange);
	thrust::copy(q.whereStartRange.begin(), q.whereStartRange.end(), cudaQuery.d_WhereStartRange);

	QueryResult cudaResult = ProcessQuerry(cube, cudaQuery);

	PrintQuerryResult(cudaResult);

	IR::QueryResult result;
	result.resize(cudaResult.ResultsCount * cudaResult.MeasPerResult);

	std::cout << cudaResult.ResultsCount << " " << cudaResult.MeasPerResult << std::endl;

	thrust::copy_n(cudaResult.d_ResultMeas, result.size(), (int64_t*) result.data());

	FreeQuerry(cudaQuery);
	FreeResult(cudaResult);

	return result;
}
