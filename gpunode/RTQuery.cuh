#ifndef RTQUERY_H_
#define RTQUERY_H_

#define RTCUBE_OP_NONE 0
#define RTCUBE_OP_SUM 1
#define RTCUBE_OP_MAX 2
#define RTCUBE_OP_MIN 3
#define RTCUBE_OP_AVG 4
#define RTCUBE_OP_CNT 5

#define RTCUBE_WHERE_NONE 0
#define RTCUBE_WHERE_SET 1
#define RTCUBE_WHERE_RANGE 2
#define RTCUBE_WHERE_MAXRANGE 3

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

typedef struct Querry
{
	int DimCount;
	int MeasCount;

	thrust::device_ptr<int> d_SelectDims;

	thrust::device_ptr<int> d_WhereDimMode;

	thrust::device_ptr<int> d_WhereDimValuesCounts;

	thrust::device_ptr<int> d_WhereDimVals;

	thrust::device_ptr<int> d_WhereDimValsStart;

	thrust::device_ptr<int> d_WhereStartRange;
	thrust::device_ptr<int> d_WhereEndRange;

	int OperationsCount;
	thrust::device_ptr<int> OperationsTypes;
	thrust::device_ptr<int> OperationsMeasures;

}Querry;

Querry InitQuerry(int dimCount, int measCount, int operationsCount);

void PrintQuerry(Querry querry);

void FreeQuerry(Querry querry);

typedef struct QueryResult
{
	Querry Q;

	int ResultsCount;

	int MeasPerResult;

	thrust::device_ptr<int> d_SelectDimSizes;

	thrust::device_ptr<int> d_ResultMeas;

}QueryResult;

void PrintQuerryResult(QueryResult result);

std::string GetQuerryResultString(QueryResult result);

void FreeResult(QueryResult result);

#endif /* RTQUERY_H_ */
