#include "RTQuery.cuh"
#include <iostream>
#include <sstream>
#include <iomanip>

Querry InitQuerry(int dimCount, int measCount, int operationsCount)
{
	Querry querry;
	querry.DimCount = dimCount;
	querry.MeasCount = measCount;

	querry.d_SelectDims = thrust::device_malloc<int>(dimCount);
	thrust::fill_n(querry.d_SelectDims, dimCount, 0);

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

	querry.d_WhereDimValsStart = thrust::device_malloc<int>(dimCount);

	querry.OperationsCount = operationsCount;
	querry.OperationsTypes = thrust::device_malloc<int>(operationsCount);
	querry.OperationsMeasures = thrust::device_malloc<int>(operationsCount);

	return querry;
}

void FreeQuerry(Querry querry)
{
	thrust::device_free(querry.OperationsMeasures);
	thrust::device_free(querry.OperationsTypes);

	thrust::device_free(querry.d_SelectDims);
	thrust::device_free(querry.d_WhereDimMode);
	thrust::device_free(querry.d_WhereDimVals);
	thrust::device_free(querry.d_WhereDimValsStart);
	thrust::device_free(querry.d_WhereDimValuesCounts);
	thrust::device_free(querry.d_WhereEndRange);
	thrust::device_free(querry.d_WhereStartRange);
}

void PrintQuerry(Querry q)
{
	printf("SELECT ");
	for (int i = 0; i < q.DimCount; ++i)
		if ((int)q.d_SelectDims[i] != 0)
			printf("d%d ", i);

	//for (int i = 0; i < q.MeasCount; ++i)
	//	if ((int)q.d_SelectMeasOps[i] != RTCUBE_OP_NONE)
	//	{
	//		switch ((int)q.d_SelectMeasOps[i])
	//		{
	//		case RTCUBE_OP_SUM:
	//			printf("SUM(m%d) ", i);
	//			break;
	//		case RTCUBE_OP_MAX:
	//			printf("MAX(m%d) ", i);
	//			break;
	//		case RTCUBE_OP_MIN:
	//			printf("MIN(m%d) ", i);
	//			break;
	//		case RTCUBE_OP_AVG:
	//			printf("AVG(m%d) ", i);
	//			break;
	//		case RTCUBE_OP_CNT:
	//			printf("CNT(m%d) ", i);
	//			break;
	//		default:
	//			printf("ERROR - UNKNOWN OP\n");
	//			return;
	//		}
	//	}

	for (int i = 0; i < q.OperationsCount; ++i)
	{
		switch ((int)q.OperationsTypes[i])
		{
		case RTCUBE_OP_SUM:
			printf("SUM(m%d) ", (int)q.OperationsMeasures[i]);
			break;
		case RTCUBE_OP_MAX:
			printf("MAX(m%d) ", (int)q.OperationsMeasures[i]);
			break;
		case RTCUBE_OP_MIN:
			printf("MIN(m%d) ", (int)q.OperationsMeasures[i]);
			break;
		case RTCUBE_OP_AVG:
			printf("AVG(m%d) ", (int)q.OperationsMeasures[i]);
			break;
		case RTCUBE_OP_CNT:
			printf("CNT(m%d) ", (int)q.OperationsMeasures[i]);
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
			printf("%10d", (int)result.d_ResultMeas[j * result.ResultsCount + i]);

		printf("\n");
	}
}

std::string GetQuerryResultString(QueryResult result)
{
	std::stringstream ss;
	ss << "Querry result\nResult lines count:" << std::setw(4) << result.ResultsCount << "\nMeas per result:" << std::setw(4) << result.MeasPerResult << "\n\n";

	for (int i = 0; i < result.ResultsCount; ++i)
	{
		int index = i;

		ss << std::setw(3) << i << ": ";

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

				ss << std::setw(3) << dimVal;

				index %= result.d_SelectDimSizes[j];
			}
		}

		ss << "  :";

		for (int j = 0; j < result.MeasPerResult; ++j)
			ss << std::setw(10) << (int)result.d_ResultMeas[j * result.ResultsCount + i];

		ss << "\n";
	}
	return ss.str();
}

void FreeResult(QueryResult result)
{
	thrust::device_free(result.d_ResultMeas);
	thrust::device_free(result.d_SelectDimSizes);
}
