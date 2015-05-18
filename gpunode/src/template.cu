////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes CUDA
#include <cuda_runtime.h>

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel(float *g_idata, float *g_odata)
{
    // access thread id
    const unsigned int tid = threadIdx.x;
    // access number of threads in this block
    const unsigned int num_threads = blockDim.x;

    // num_threads -- polowa dlugosci tablicy, potega 2
    unsigned int step = 1;
    float value1;
    float value2;
    while (step != num_threads * 2) {
    	if (tid < num_threads / step) {
    		value1 = g_idata[tid * 2];
    		value2 = g_idata[tid * 2 + 1];
    	}
        __syncthreads();
    	if (tid < num_threads / step) {
    		g_idata[tid] = value1 + value2;
    	}
        __syncthreads();
    	step *= 2;
    }
    g_odata[0] = g_idata[0];
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
    bool bTestResult = true;

    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    int devID = findCudaDevice(argc, (const char **)argv);

    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkStartTimer(&timer);

    unsigned int num_threads = 16;
    unsigned int mem_size = sizeof(float) * num_threads * 2;

    // allocate host memory
    float *h_idata = (float *) malloc(mem_size);

    // initalize the memory
    for (unsigned int i = 0; i < num_threads * 2; ++i)
    {
        h_idata[i] = (float) i;
    }

    // allocate device memory
    float *d_idata;
    checkCudaErrors(cudaMalloc((void **) &d_idata, mem_size));
    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // allocate device memory for result
    float *d_odata;
    checkCudaErrors(cudaMalloc((void **) &d_odata, mem_size));
    // copy host memory to device
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, mem_size, cudaMemcpyHostToDevice));

    // setup execution parameters
    dim3  grid(1, 1, 1);
    dim3  threads(num_threads, 1, 1);

    // execute the kernel
    testKernel<<< grid, threads, mem_size >>>(d_idata, d_odata);

    // check if kernel execution generated and error
    getLastCudaError("Kernel execution failed");

    // allocate mem for the result on host side
    float *h_odata = (float *) malloc(mem_size);
    // copy result from device to host
    checkCudaErrors(cudaMemcpy(h_odata, d_odata, mem_size, cudaMemcpyDeviceToHost));

    printf("Hodata: %f\n\n", h_odata[0]);

    sdkStopTimer(&timer);
    printf("Processing time: %f (ms)\n", sdkGetTimerValue(&timer));
    sdkDeleteTimer(&timer);

    // compute reference solution
    float *reference = (float *) malloc(mem_size);
    computeGold(reference, h_idata, num_threads);

    // check result
    if (checkCmdLineFlag(argc, (const char **) argv, "regression"))
    {
        // write file for regression test
        sdkWriteFile("./data/regression.dat", h_odata, num_threads, 0.0f, false);
    }
    else
    {
        // custom output handling when no regression test running
        // in this case check if the result is equivalent to the expected soluion
        bTestResult = compareData(reference, h_odata, num_threads, 0.0f, 0.0f);
    }

    // cleanup memory
    free(h_idata);
    free(h_odata);
    free(reference);
    checkCudaErrors(cudaFree(d_idata));
    checkCudaErrors(cudaFree(d_odata));

    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits   
    cudaDeviceReset();
    exit(bTestResult ? EXIT_SUCCESS : EXIT_FAILURE);
}
