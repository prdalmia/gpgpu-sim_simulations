#include <cuda.h>
#include <stdio.h>
#include "randoms.h"
#include <math.h>
#include <time.h>
#include<iostream>
#include<string.h>
#include<stdlib.h>

using namespace std;

    
 __global__ void kernel_getHist(int* array, long size,  int* histo, int buckets)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if(tid>=size)   return;

     int value = array[tid];

    int bin = value % buckets;

    atomicAdd(&histo[bin],1);
    //__syncthreads();
}

  __global__ void CalHistKernel(int* array , long size,  int* histo ,int buckets)
{
extern __shared__ int _bins[];

int tx = threadIdx.x;
int idx = blockIdx.x*blockDim.x+threadIdx.x;//blockDim.y=1

if(tx< blockDim.x)
{

    for (int i = 0; i < buckets/blockDim.x ; i++)
     _bins[i*blockDim.x+tx]=0;     

}
__syncthreads();

 for(int i = 0; i < size/(gridDim.x*blockDim.x); i ++)
if(idx + i*blockDim.x <size)
{       
    atomicAdd((int*)&_bins[array[idx+i*(size/gridDim.x)] % buckets],1);     

}
__syncthreads();
for (int i = 0; i < buckets/blockDim.x; i++)
atomicAdd((int*)&histo[i*blockDim.x+tx],_bins[i*blockDim.x+tx]);
}

void histogram256CPU(int *h_Histogram, int *histo, long long size, int buckets)
 {

     for (long long i = 0; i < buckets ; i++)
     {

          histo[i] = 0;

     }

     printf("\n \n Running CPU Function \n \n ");
     for (long long i = 0; i < size ; i++)
     {
        int bin = h_Histogram[i] % buckets;

          histo[bin]++;

     }


 }




int main(int argc, char *argv[]) {
    
    int buckets = 256;
    int seed = 1;
    
    long size = 3840 * 2160;      // 4k
   // long size = 2048*1080;    // 4k
  // long size = 512;
   int num_blocks = 80*16;
    const int threadsPerBlock = 256;
    int *hA = new int[size];
    int *hB = new int[buckets];
    int *histo = new int[buckets];
    
    random_ints(hA, 0, 10000, size, seed);
    random_ints(hB, 0, 0,  buckets, seed); 
/*

     for (long long i = 0; i < size ; i++)
     {
               hA[i] = 2;
               
     }
*/
    
    
       int* dArray;
    cudaMalloc(&dArray,size * sizeof(int));
    cudaMemcpy(dArray,hA,size * sizeof(int) ,cudaMemcpyHostToDevice);

     int* dHist;
    cudaMalloc(&dHist,buckets * sizeof(int));
    cudaMemset(dHist,0,buckets * sizeof(int));

    dim3 block(threadsPerBlock);
    dim3 grid((size + block.x - 1)/block.x);

  // kernel_getHist<<<grid,block>>>(dArray,size,dHist,buckets);
   CalHistKernel<<<num_blocks,block, buckets * sizeof(int)>>>(dArray,size,dHist,buckets);
   
   cudaDeviceSynchronize();
    cudaMemcpy(histo,dHist,buckets * sizeof(int),cudaMemcpyDeviceToHost);

    histogram256CPU(hA, hB, size, buckets);
   
   
   

    
    printf("\n Final histogram Value \n \n");
    
    for (int j = 0 ; j < buckets; j++){
        //printf(" %d, ", hB[j]);
        if (histo[j] != hB[j])
        { 
            printf(" \n \n , ");
            printf("Failed at index = %d \n", j);
            printf("CPU value = %d \t", hB[j]);
            printf("GPU value = %d \n", histo[j]);
            //return 0;
        
        }
    }
            
    printf("\n Passed \n");


   cudaFree(dArray);
   cudaFree(dHist);
   delete[] hA;
   delete[] hB;
   delete[] histo ;

   
    
    return 0;
    
}

