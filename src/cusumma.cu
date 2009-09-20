#include <cuda.h>
#include <cublas.h>
#include <math.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

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
  float *hA, *dA, *dB, *dC;
  int i, j, diff, tm, tk, tp, tp_last, tmp1, tmp2;
  char opA, opB;
  float factor;
  unsigned int gpu_mem, _m, _mmax, _moff, _k, _kmax, _koff;

  cublasInit();

  // get total free memory available to SUMMA
  cuMemGetInfo(&gpu_mem, NULL);

  // take 2MB off the top for CUBLAS working memory
  // this is a guess that seems to work, replace with actual numbers when known
  gpu_mem -= 2*1048576;

  // convert gpu_mem from bytes into matrix elements (floats) for simplicity
  gpu_mem /= sizeof(float);

  // determine optimal partition dimensions
  tp = 100000;
  tm = 0;
  do {
    if(tp > 0)
      tp_last = tp;
    _mmax = ceil(1.0*m/++tm);
    tmp1 = A == B ? gpu_mem - _mmax * _mmax : gpu_mem - n * _mmax;
    tmp2 = A == B ? _mmax : n + _mmax;
    _kmax = tmp1 / tmp2; //(gpu_mem - n * _mmax)/(n + _mmax);
    tk    = ceil(1.0*k/_kmax);
    tp    = (A == A ? 1 : 2)*tm*tk + tm;
  } while(tp < 0 || tp < tp_last);

  _mmax = ceil(1.0*m/--tm);
  if(A == B) {
    _kmax = gpu_mem / _mmax - _mmax;
  } else {
    _kmax = ( gpu_mem - _mmax * n ) / ( _mmax + n );
  }
 
//_mmax = 2;
//_kmax = 2;

  // assumes input matrices are in row-major order
  opA = transB ? 't' : 'n';
  opB = transA ? 't' : 'n';

  _m    = _mmax;
  _moff = 0;
  while(_moff < m) {
    cublasAlloc(_m * n, sizeof(float), (void**)&dC);
    if(A == B) {  // op(A) * op(A)

      diff = gpu_mem - m*k - m*m;
      // A can fit entirely on the device
      if((_m == m) && (diff > 0)) {
        cublasAlloc(m * k, sizeof(float), (void**)&dA);
        cublasSetVector(m * k, sizeof(float), A, 1, dA, 1);
        
        cublasSgemm(opA, opB, m, m, k, 1.0f, dA, k, dA, k, 0.0f, dC, m);
        cublasFree(dA);

      } else {
        _koff  = 0;
        _k     = _kmax;
        factor = 0.0f;
        while(_koff < k) {
          cublasAlloc(_m * _k, sizeof(float), (void**)&dA);

          hA = (float*)malloc(_m * _k * sizeof(float));
          for(i = 0; i < _m; ++i)
            for(j = 0; j < _k; ++j)
              hA[i*_k + j] = A[(i+_moff)*k + j + _koff];
          cublasSetVector(_m * _k, sizeof(float), hA, 1, dA, 1);
          free(hA);

          cublasSgemm(opA, opB, _m, _m, _k, 1.0f, dA, _k, dA, _k, factor, dC, _m);
          cublasFree(dA);
        
          _koff += _k;
          _k     = k - _koff > _kmax ? _kmax : k - _koff;
          factor = 1.0f;
        }
      }

    } else { // op(A) * op(B)
      cublasAlloc(_m * n, sizeof(float), (void**)&dC);
      diff = gpu_mem - (m*k + k*n + m*n);
      if((_m == m) && (diff > 0)) {
        cublasAlloc(m * k, sizeof(float), (void**)&dA);
        cublasSetVector(m * k, sizeof(float), A, 1, dA, 1);

        cublasAlloc(k * n, sizeof(float), (void**)&dB);
        cublasSetVector(k * n, sizeof(float), B, 1, dB, 1);
    
        cublasSgemm(opA, opB, n, m, k, 1.0f, dB, n, dA, k, 0.0f, dC, m);

        cublasFree(dA);
        cublasFree(dB);

      } else {
        _koff  = 0;
        _k     = _kmax;
        factor = 0.0f;
        while(_koff < k) {
          cublasAlloc(_m * _k, sizeof(float), (void**)&dA);
          cublasAlloc(_k * n, sizeof(float), (void**)&dB);

          hA = (float*) malloc(_m * _k * sizeof(float));
          for(i = 0; i < _m; ++i)
            for(j = 0; j < _k; ++j)
              hA[i*_k + j] = A[(i+_moff)*k + j + _koff];
          cublasSetVector(_m * _k, sizeof(float), hA, 1, dA, 1);
          free(hA);
/*
        //hB = (float*) malloc(_k * n * sizeof(float));
        for(i = 0; i < _k; ++i)
          for(j = 0; j < n; ++j)
            hB[i*n + j] = B[(i+_koff)*n + j];
        //cublasSetMatrix(_k, n, sizeof(float), hB, _k, dB, _k);
        cublasSetVector(_k * n, sizeof(float), hB, 1, dB, 1);
        free(hB);
*/
          int ldb = transB ? _k : n;
          cublasSetVector(_k * n, sizeof(float), B+(n*_koff), 1, dB, 1);
          cublasSgemm(opA, opB, n, _m, _k, 1.0f, dB, ldb, dA, _k, factor, dC, n);
          cublasFree(dA);
          cublasFree(dB);

          _koff += _k;
          _k = k - _koff > _kmax ? _kmax : k - _koff;
          factor = 1.0f;
        }
      }
    }

    cublasGetVector(_m*n, sizeof(float), dC, 1, C+(_moff*n), 1);
    cublasFree(dC);

    _moff += _m;
    _m = m - _moff > _mmax ? _mmax : m - _moff;
  }

  cublasShutdown();
}

int main(int argc, char** argv) {
  struct timeval start, end; 
  double elapsed;

  float *A, *B, *C;
  int i, m, n, k;
 
  m = 500;
  n = 500;
  k = 400000;
  A = (float*)malloc(m*k*sizeof(float));
  B = (float*)malloc(k*n*sizeof(float));
  C = (float*)calloc(m*n,sizeof(float));

  for(i = 0; i < m*k; ++i)
    A[i] = 1;
  for(i = 0; i < k*n; ++i)
    B[i] = 1;

for(i=0;i<11;++i) {
  gettimeofday(&start,NULL);
  cusumma(0,1,m,n,k,A,B,C);
  gettimeofday(&end,NULL);

  elapsed = ((end.tv_sec*1000000 + end.tv_usec) - (start.tv_sec*1000000 + start.tv_usec))/1000000.0;
  printf("%f %f %f %f\n", C[0], C[m-1], C[m*(n-1)], C[m*n-1]);
//  for(i = 0; i < m*m; ++i)
//    printf("%f\n",C[i]);
  printf("%f\n", elapsed);
}

  free(A);
  free(B);
  free(C);
}
