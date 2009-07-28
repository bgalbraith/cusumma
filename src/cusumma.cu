#include <cuda.h>
#include <cublas.h>

extern "C" void
cudaSUMMA(unsigned int transA, 
          unsigned int transB, 
          unsigned int m,
          unsigned int n,
          unsigned int k,
          float *A, 
          float *B, 
          float *C)
{
  float *hA, *hB, *hC, *dA, *dB, *dC;
  int i, j, diff, offset, kmax;
  char opA, opB;
  unsigned int gpu_mem;
  size_t a_mem, b_mem, col_mem, row_mem;

  cublasInit();

  // get total free memory available to SUMMA
  cuMemGetInfo(&gpu_mem, NULL);

  // take 1MB off the top for CUBLAS working memory
  // this is a guess that seems to work, replace with actual numbers when known
  gpu_mem -= 1048576;

  // allocate and initialize result matrix, substract from total free memory
  cublasAlloc(m * n, sizeof(float), (void**)&dC);
  cublasSetVector(m * n, sizeof(float), C, 1, dC, 1); 
  gpu_mem -= m * n * sizeof(float);

  // assumes input matrices are in row-major order
  opA = transA ? 'n' : 't';
  opB = transB ? 'n' : 't';

  // op(A) * op(A)
  if(A == B) { 
    a_mem = A->size * sizeof(float);
    diff = gpu_mem - a_mem;
    // A can fit entirely on the device
    if(diff > 0) {
      cublasAlloc(m * k, sizeof(float), (void**)&dA);
      cublasSetVector(m * k, sizeof(float), A, 1, dA, 1);
        
      k = transA ? A->rows : A->cols;
      cublasSgemm(opA, opB, C->rows, C->cols, k, 1.0f, dA, A->cols, dA, A->cols, 0.0f, dC, C->rows);
      cublasFree(dA);

    } else {
      // tk: assume cols for now, if transA, will have to flip
      col_mem = A->rows * sizeof(float);
      kmax    = gpu_mem / col_mem;
      k       = kmax;
      offset  = 0;

      while(offset < A->cols) {
        hA = (float*) malloc(A->rows * k * sizeof(float));
        cublasAlloc(A->rows * k, sizeof(float), (void**)&dA);

        for(i = 0; i < A->rows; ++i)
          for(j = 0; j < k; ++j)
            hA[i*k + j] = A->data[i*A->cols + j + offset];
        cublasSetVector(A->rows * k, sizeof(float), hA, 1, dA, 1);
        free(hA);

        cublasSgemm(opA, opB, C->rows, C->cols, k, 1.0f, dA, k, dA, k, 1.0f, dC, C->rows);
        cublasFree(dA);
        
        offset += k;
        k = A->cols - offset > kmax ? kmax : A->cols - offset;
      }
    }

  } else {
    a_mem = A->size * sizeof(float);
    b_mem = B->size * sizeof(float);
    diff = gpu_mem - a_mem - b_mem;
    if(diff > 0) {

      cublasAlloc(A->size, sizeof(float), (void**)&dA);
      cublasSetVector(A->size, sizeof(float), A->data, 1, dA, 1);

      cublasAlloc(B->size, sizeof(float), (void**)&dB);
      cublasSetVector(B->size, sizeof(float), B->data, 1, dB, 1);
      
      k = transA ? A->rows : A->cols;
      cublasSgemm(opA, opB, C->rows, C->cols, k, 1.0f, dA, A->cols, dB, B->cols, 1.0f, dC, C->rows);

      cublasFree(dA);
      cublasFree(dB);

    } else {
      // tk: handle transpose, currently assumes A * B'
      col_mem = A->rows * sizeof(float);
      row_mem = B->rows * sizeof(float);
      kmax    = gpu_mem / (col_mem + row_mem);
      k       = kmax;
      offset  = 0;

      while(offset < A->cols) {
        hA = (float*) malloc(A->rows * k * sizeof(float));
        cublasAlloc(A->rows * k, sizeof(float), (void**)&dA);
        hB = (float*) malloc(k * B->rows * sizeof(float));
        cublasAlloc(k * B->rows, sizeof(float), (void**)&dB);
 

        for(i = 0; i < A->rows; ++i)
          for(j = 0; j < k; ++j)
            hA[i*k + j] = A->data[i*A->cols + j + offset];
        cublasSetVector(A->rows * k, sizeof(float), hA, 1, dA, 1);
        free(hA);

        for(i = 0; i < B->rows; ++i)
          for(j = 0; j < k; ++j)
            hB[i*k + j] = B->data[i*B->cols + j + offset];
        cublasSetVector(k * B->rows, sizeof(float), hB, 1, dB, 1);
        free(hB);

        cublasSgemm(opA, opB, C->rows, C->cols, k, 1.0f, dA, k, dB, k, 1.0f, dC, C->rows);
        cublasFree(dA);
        cublasFree(dB);
        
        offset += k;
        k = A->cols - offset > kmax ? kmax : A->cols - offset;
      }
    }
  }

  hC = (float*) calloc(C->size, sizeof(float));
  cublasGetVector(C->size, sizeof(float), dC, 1, hC, 1);
  cublasFree(dC);

  for(i = 0; i < C->rows; ++i)
    for(j = 0; j < C->cols; ++j)
      C->data[i*C->cols+j] = hC[j*C->rows+i];
  free(hC);  

  cublasShutdown();
}
