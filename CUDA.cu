#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "get_walltime.c"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"%s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void array_print(float *arr, int length) 
{
  for (int i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}



__global__ void bitonicSort(float *GPU_array, int j, int i)
{
  unsigned int index, swap_index; 
  index = threadIdx.x + blockDim.x * blockIdx.x;
  swap_index = index^j;

  if ((swap_index)>index) {
    if ((index&i)==0 && GPU_array[index]>GPU_array[swap_index]) {
      float temp = GPU_array[index];
      GPU_array[index] = GPU_array[swap_index];
      GPU_array[swap_index] = temp;
    }
    if ((index&i)!=0 && GPU_array[index]<GPU_array[swap_index]) {
      float temp = GPU_array[index];
      GPU_array[index] = GPU_array[swap_index];
      GPU_array[swap_index] = temp;
    }
  }
}

int main(void)
{
    int N;
    double start=0, stop=0;
    
    scanf("%d", &N);

    int num_threads = 32;
    int num_blocks = (N+num_threads-1)+num_threads;
    float *array = new float[N];
    float *GPU_array;
    size_t GPU_N = N * sizeof(float);

    srand(time(NULL));
    float a = 10.0;

    for (int i = 0; i < N; i++){
      array[i] = float((rand())/float((RAND_MAX)) * a);
    }

    //array_print(array, N);

    get_walltime(&start); 
    
    cudaMalloc((void**) &GPU_array, GPU_N);
    gpuErrchk( cudaMemcpy(GPU_array, array, GPU_N, cudaMemcpyHostToDevice) );
    cudaMemcpy(GPU_array, array, GPU_N, cudaMemcpyHostToDevice);
    
    

    dim3 blocks(num_blocks,1);    
    dim3 threads(num_threads,1);  

    
    for (int i = 2; i <= N; i *= 2) {
      for (int j=i>>1; j>0; j=j>>1) {
        bitonicSort<<<blocks, threads>>>(GPU_array, j, i);
        cudaDeviceSynchronize();
      }
    }
    
    cudaMemcpy(array, GPU_array, GPU_N, cudaMemcpyDeviceToHost);
    
    cudaFree(GPU_array);

    //array_print(array, N);
    get_walltime(&stop);

    
    printf("Time taken for array size %d with %d threads: %.3fs\n", N, num_threads, stop-start);
 
}