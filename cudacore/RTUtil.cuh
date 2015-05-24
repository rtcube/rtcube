#ifndef RTUTIL_H_
#define RTUTIL_H_

#include <thrust/device_vector.h>

void PrintVector(int dimCount, int *dimVals, int measCount, int *measVals);
void PrintPack(int vecCount, int dimCount, int measCount, int **dims, int **meas);
void GeneretVector(int dimCount, int *dimRanges, int measCount, int measMax, int **dimVals, int **measVals);
void GeneratePack(int vecCount, int dimCount, int *dimRanges, int measCount, int measMax, int ***dims, int ***meas);
void FreePack(int vecCount, int ***dims, int ***meas);
void PrintPackedPack(int vecCount, int dimCount, int *dims, int measCount, int *meas);
void PrepareDataForInsert(int vecCount, int dimCount, int **dims, int measCount, int **meas, thrust::device_ptr<int> *d_dimsPacked, thrust::device_ptr<int> *d_measPacked);

void RunSample();

#endif /* RTUTIL_H_ */
