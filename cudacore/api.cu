#include "api.h"

#include "RTCube.cuh"
#include <vector>

namespace CudaCore
{
	struct RTCubeP
	{
		::RTCube cube;
	};

	RTCube::RTCube(const IR::CubeDef& cubeDef)
		: p(new RTCubeP())
	{
		int *dimensionsSizes = (int*)malloc(sizeof(int)*cubeDef.dims.size());
		for(int i = 0; i < cubeDef.dims.size(); ++i)
			dimensionsSizes[i] = cubeDef.dims[i].range;

		p->cube = InitCube(MEM_TO_FILL, cubeDef.dims.size(), dimensionsSizes, cubeDef.meas.size(), BLOCKS, THREADS);
		free(dimensionsSizes);

		//PrintCubeInfo(p->cube);
	}

	RTCube::~RTCube()
	{
		FreeCube(p->cube);
		delete p;
	}

	void RTCube::insert(const IR::Rows& rows)
	{
		thrust::device_ptr<int> DimensionsPacked = thrust::device_malloc<int>(rows.num_dims * rows.num_rows);
		thrust::device_ptr<int> MeasuresPacked = thrust::device_malloc<int>(rows.num_meas * rows.num_rows);

		std::vector<int> hostDimenisons;
		std::vector<float> hostMeasures;

		for(int i = 0; i < rows.num_dims; ++i)
			for(int j = 0; j < rows.num_rows; ++j)
				hostDimenisons.push_back(rows.dims[j * rows.num_dims + i]);

		for(int i = 0; i < rows.num_meas; ++i)
			for(int j = 0; j < rows.num_rows; ++j)
				hostMeasures.push_back(rows.meas[j * rows.num_meas + i].f);

		thrust::copy_n(hostDimenisons.begin(), rows.num_dims * rows.num_rows, DimensionsPacked);
		thrust::copy_n(hostMeasures.begin(), rows.num_meas * rows.num_rows, MeasuresPacked);

		AddPack(p->cube, rows.num_rows, DimensionsPacked, MeasuresPacked);

		thrust::device_free(DimensionsPacked);
		thrust::device_free(MeasuresPacked);

		//PrintCubeInfo(p->cube);
		//PrintCubeMemory(p->cube);

	}

	IR::QueryResult RTCube::query(const IR::Query& q)
	{
		Querry cudaQuery = InitQuerry(q.DimCount, q.MeasCount, q.operationsCount);

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

		QueryResult cudaResult = ProcessQuerry(p->cube, cudaQuery);

		//PrintQuerryResult(cudaResult);

		IR::QueryResult result(q);

		thrust::copy_n(cudaResult.d_ResultMeas, result.resultMeas.size(), result.resultMeas.begin());
		thrust::copy_n(cudaResult.d_SelectDimSizes, result.selectDimSizes.size(), result.selectDimSizes.begin());

		FreeQuerry(cudaQuery);
		FreeResult(cudaResult);

		return result;
	}
}
