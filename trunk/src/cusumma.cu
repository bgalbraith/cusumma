#include <cuda.h>
#include <cublas.h>
#include <math.h>
#include <stdio.h>

extern "C" void
cusumma(unsigned int transA, 
        unsigned int transB, 
        unsigned int m,
        unsigned int n,
        unsigned int k,
        float *A, 
        float *B, 
        float *C)
{
  float *hA, *hB, *hC, *dA, *dB, *dC;
  float m_opt, k_opt;
  int i, j, diff, offset, _kmax, _m, _k, tm, tk;
  char opA, opB;
  unsigned int gpu_mem;

  cublasInit();

  // get total free memory available to SUMMA
  cuMemGetInfo(&gpu_mem, NULL);

  // take 2MB off the top for CUBLAS working memory
  // this is a guess that seems to work, replace with actual numbers when known
  gpu_mem -= 2*1048576;

  // convert gpu_mem from bytes into matrix elements (floats) for simplicity
  gpu_mem /= sizeof(float);

/*
  // determine optimal partition configuration
  // assume C is whole
  tw    = ceil((m*k + k*n)/(1.0*s - m*n));
  // assume C is partitioned
  m_opt = (sqrt(4.0*m*k*gpu_mem + (k+m)*(k+m)*n*n) - (k+m)*n)/(2.0*k);
  k_opt = (1.0*k/m) * m_opt;
  _m    = floor(m_opt);
  _k    = floor(k_opt);
  tk    = ceil(1.0*k/_k);
  tm    = ceil(1.0*m/_m);
  tp    = tk + tk*tm + tm;

  plan  = (tw > 0 && tw < tp) ? SINGLE_PARTITION : DOUBLE_PARTITION;
*/

  // assumes input matrices are in row-major order
  opA = transA ? 'n' : 't';
  opB = transB ? 'n' : 't';

  // op(A) * op(A)
  if(A == B) { 
    // allocate and initialize result matrix, substract from total free memory
    cublasAlloc(m * n, sizeof(float), (void**)&dC);
    cublasSetVector(m * n, sizeof(float), C, 1, dC, 1); 
    gpu_mem -= m * n;

    diff = gpu_mem - m*k;
    // A can fit entirely on the device
    if(diff > 0) {
      cublasAlloc(m * k, sizeof(float), (void**)&dA);
      cublasSetVector(m * k, sizeof(float), A, 1, dA, 1);
        
      cublasSgemm(opA, opB, m, n, k, 1.0f, dA, k, dA, k, 0.0f, dC, m);
      cublasFree(dA);

    } else {
      // tk: assume cols for now, if transA, will have to flip
      _kmax  = gpu_mem / m;

      offset = 0;
      _k     = _kmax;
      while(offset < k) {
        hA = (float*)malloc(m * _k * sizeof(float));
        cublasAlloc(m * _k, sizeof(float), (void**)&dA);

        for(i = 0; i < m; ++i)
          for(j = 0; j < _k; ++j)
            hA[i * _k + j] = A[i * k + j + offset];
        cublasSetVector(m * _k, sizeof(float), hA, 1, dA, 1);
        free(hA);

        cublasSgemm(opA, opB, m, n, _k, 1.0f, dA, _k, dA, _k, 1.0f, dC, m);
        cublasFree(dA);
        
        offset += _k;
        _k      = k - offset > _kmax ? _kmax : k - offset;
      }
    }
  } 
  // op(A) * op(B)
  else {
    cublasAlloc(m * n, sizeof(float), (void**)&dC);
    cublasSetVector(m * n, sizeof(float), C, 1, dC, 1); 
    gpu_mem -= m * n;

    diff = gpu_mem - (m*k + k*n);
    if(diff > 0) {

      cublasAlloc(m * k, sizeof(float), (void**)&dA);
      cublasSetVector(m * k, sizeof(float), A, 1, dA, 1);

      cublasAlloc(k * n, sizeof(float), (void**)&dB);
      cublasSetVector(k * n, sizeof(float), B, 1, dB, 1);
    
      cublasSgemm(opA, opB, m, n, k, 1.0f, dA, k, dB, k, 0.0f, dC, m);

      cublasFree(dA);
      cublasFree(dB);

    } else {

      // tk: handle transpose, currently assumes A * B'
      _kmax   = gpu_mem / (m + n);
      _k      = _kmax;
      offset  = 0;

      while(offset < k) {
        hA = (float*) malloc(m * _k * sizeof(float));
        cublasAlloc(m * _k, sizeof(float), (void**)&dA);
        hB = (float*) malloc(_k * n * sizeof(float));
        cublasAlloc(_k * n, sizeof(float), (void**)&dB);
 

        for(i = 0; i < m; ++i)
          for(j = 0; j < _k; ++j)
            hA[i*_k + j] = A[i*k + j + offset];
        cublasSetVector(m * _k, sizeof(float), hA, 1, dA, 1);
        free(hA);

        for(i = 0; i < n; ++i)
          for(j = 0; j < _k; ++j)
            hB[i*_k + j] = B[i*n + j + offset];
        cublasSetVector(_k * n, sizeof(float), hB, 1, dB, 1);
        free(hB);

        cublasSgemm(opA, opB, m, n, _k, 1.0f, dA, _k, dB, _k, 1.0f, dC, m);
        cublasFree(dA);
        cublasFree(dB);
        
        offset += _k;
        _k = k - offset > _kmax ? _kmax : k - offset;
      }
    }
  }

  hC = (float*) calloc(m*n, sizeof(float));
  cublasGetVector(m*n, sizeof(float), dC, 1, hC, 1);
  cublasFree(dC);

  for(i = 0; i < m; ++i)
    for(j = 0; j < n; ++j)
      C[i*n+j] = hC[j*m+i];
  free(hC);  

  cublasShutdown();
}

int main(int argc, char** argv) {
  float *A, *B, *C;
  int i, m, n, k;
 
  m = 1000;
  n = 2;
  k = 400000;
  A = (float*)malloc(m*k*sizeof(float));
  B = (float*)malloc(k*n*sizeof(float));
  C = (float*)malloc(m*n*sizeof(float));

  for(i = 0; i < m*k; ++i)
    A[i] = 1;
  for(i = 0; i < k*n; ++i)
    B[i] = 1;

  // tk: trans(A)*A doesn't work if A isn't square
  //     need to swap params inside routine, not here
  //     maybe just go with full cblas-style inputs
  cusumma(0,1,m,n,k,A,B,C);
  printf("%f %f %f\n", C[0], C[1], C[2]);
  free(A);
  free(B);
  free(C);
}
