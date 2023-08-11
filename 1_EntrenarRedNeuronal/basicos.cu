#include "basicos.cuh"

void manageCUDAError(cudaError_t status) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Error de CUDA: %s\n", cudaGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

curandGenerator_t crearGeneradorNumerosAleatoriosEnDistribucionNormal() {
    curandGenerator_t curandGenerator;
    curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_MT19937);
    curandSetGeneratorOrdering(curandGenerator, CURAND_ORDERING_PSEUDO_BEST);
    manageCUDAError(cudaDeviceSynchronize());
    return curandGenerator;
}

void generarNumerosAleatoriosEnDistribucionNormal(curandGenerator_t curandGenerator, float mean, float sdev, float* pointer, long long nelems) {
    unsigned long long semilla = rand() % 10000;
    curandSetPseudoRandomGeneratorSeed(curandGenerator, semilla);
    curandGenerateNormal(curandGenerator, (float*)pointer, nelems, mean, sdev);
    manageCUDAError(cudaDeviceSynchronize());
}

//FORWARD

// A* B = C, A = MxN, B = NxP, C = MxP
__global__ void productoMatrices(const float* A, const float* B, float* C, int M, int N, int P) {

    // Shared memory for the tiles of matrices A and B
    __shared__ float tileA[32][32];
    __shared__ float tileB[32][32];

    // Thread and block indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Row and column in the output matrix C
    int row = by * blockDim.y + ty;
    int col = bx * blockDim.x + tx;

    float sum = 0.0f;

    // Loop over tiles of matrices A and B
    for (int t = 0; t < (N + 32 - 1) / 32; ++t) {
        // Load tile of matrix A into shared memory
        if (row < M && t * 32 + tx < N) {
            tileA[ty][tx] = A[row * N + t * 32 + tx];
        }
        else {
            tileA[ty][tx] = 0.0f;
        }

        // Load tile of matrix B into shared memory
        if (col < P && t * 32 + ty < N) {
            tileB[ty][tx] = B[(t * 32 + ty) * P + col];
        }
        else {
            tileB[ty][tx] = 0.0f;
        }

        // Synchronize threads to ensure all data is loaded
        __syncthreads();

        // Compute the dot product of the tiles
        for (int i = 0; i < 32; ++i) {
            sum += tileA[ty][i] * tileB[i][tx];
        }

        // Synchronize threads before loading the next tiles
        __syncthreads();
    }

    // Write the result to the output matrix C
    if (row < M && col < P) {
        C[row * P + col] = sum;
    }

}

//m-> [ m[i] + v for i = numero_filas(m) ]
__global__ void sumarCadaFilaMatrizVector(float* m, float* v, int nrows, int ncols)
{
    __shared__ double tile[32][32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idy < ncols) {
        tile[threadIdx.x][threadIdx.y] = v[idy];
    }

    __syncthreads();

    if (idx < nrows && idy < ncols) {
        m[idx * ncols + idy] += tile[threadIdx.x][threadIdx.y];
    }

}

//m-> [ [ funcion_sigmoide( m[i][j] ) for j = numero_columnas(m) ] for i = numero_filas(m) ]
__global__ void aplicarFuncionSigmoideCadaElementoMatriz(float* zl, float* al, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        al[idx * ncols + idy] = 1 / (1 + expf(-zl[idx * ncols + idy]));
    }
}

__global__ void aplicarFuncionTahnCadaElementoMatriz(float* zl, float* al, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        //al[idx * ncols + idy] = (expf(zl[idx * ncols + idy]) - expf(-zl[idx * ncols + idy])) / (expf(zl[idx * ncols + idy]) + expf(-zl[idx * ncols + idy]));
        //al[idx * ncols + idy] = ( 2 / (1 + expf(-2*zl[idx * ncols + idy]) ) ) - 1;
        al[idx * ncols + idy] = tanhf(zl[idx * ncols + idy]);
    }
}

__global__ void aplicarFuncionCosenoEspecialCadaElementoMatriz(float* zl, float* al, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        al[idx * ncols + idy] = zl[idx * ncols + idy] * cosf(zl[idx * ncols + idy]);
    }
}

__global__ void aplicarFuncionReluCadaElementoMatriz(float* zl, float* al, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        al[idx * ncols + idy] = fmaxf( 0, zl[idx * ncols + idy]);
    }
}

__global__ void aplicarFuncionLinealCadaElementoMatriz(float* zl, float* al, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        al[idx * ncols + idy] = zl[idx * ncols + idy];
    }
}

__global__ void aplicarFuncionELUCadaElementoMatriz(float* zl, float* al, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        if (zl[idx * ncols + idy] < 0) { al[idx * ncols + idy] = 0.03 * ( expf(zl[idx * ncols + idy]) - 1 ); }
        else { al[idx * ncols + idy] = zl[idx * ncols + idy]; }
    }
}

//ERROR CAPA OUTPUT

__global__ void aplicarDerivadaFuncionPerdidaMSECadaElementoPredY(int batch_size, int nvalssalida, float* pred_y, float* real_y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < batch_size && idy < nvalssalida) {
        pred_y[idx * nvalssalida + idy] = 2 * (pred_y[idx * nvalssalida + idy] - real_y[idx * nvalssalida + idy]);
    }
}

__global__ void aplicarFuncionCosteMSE(int batch_size, int nvalssalida, const float* pred_y, const float* real_y, float* res) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < batch_size && idy < nvalssalida) {
        float tempres = pred_y[idx * nvalssalida + idy] - real_y[idx * nvalssalida + idy];
        res[idx * nvalssalida + idy] = tempres* tempres;
    }
}

//BACKWARD

//m-> [ [ derivada_funcion_sigmoide( m[i][j] ) for j = numero_columnas(m) ] for i = numero_filas(m) ]
__global__ void aplicarDerivadaFuncionSigmoideCadaElementoMatriz(float* m, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        float res = 1 / (1 + expf(-m[idx * ncols + idy]));
        m[idx * ncols + idy] = res * (1 - res);
    }
}

__global__ void aplicarDerivadaFuncionTahnCadaElementoMatriz(float* m, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        //float res = (expf(m[idx * ncols + idy]) - expf(-m[idx * ncols + idy])) / (expf(m[idx * ncols + idy]) + expf(-m[idx * ncols + idy]));
        //m[idx * ncols + idy] = 1 - res * res;
        //m[idx * ncols + idy] = 4*expf(2 * m[idx * ncols + idy]) / powf(1 + expf(2 * m[idx * ncols + idy]), 2.0);
        float res = tanhf(m[idx * ncols + idy]);
        m[idx * ncols + idy] = 1 - (res * res);
    }
}

__global__ void aplicarDerivadaFuncionCosenoEspecialCadaElementoMatriz(float* m, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        m[idx * ncols + idy] = -1 * sinf(m[idx * ncols + idy]);
    }
}

__global__ void aplicarDerivadaFuncionReluCadaElementoMatriz(float* m, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        if (m[idx * ncols + idy] < 0) { m[idx * ncols + idy] = 0; }
        else{ m[idx * ncols + idy] = 1; }
    }
}
__global__ void aplicarDerivadaFuncionLinealCadaElementoMatriz(float* m, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        m[idx * ncols + idy] = 1;
    }
}

__global__ void aplicarDerivadaFuncionELUCadaElementoMatriz(float* m, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        if (m[idx * ncols + idy] < 0) { m[idx * ncols + idy] = 0.03 * expf(m[idx * ncols + idy]); }
        else { m[idx * ncols + idy] = 1; }
    }
}

//T(idata) = odata
__global__ void matrizTraspuesta(float* odata, float* idata, int nrows, int ncols)
{
    __shared__ double tile[32][32];
    int i_n = blockIdx.x * 32 + threadIdx.x;
    int i_m = blockIdx.y * 32 + threadIdx.y; // <- threadIdx.y only between 0 and 7

    // Load matrix into tile
    // Every Thread loads in this case 4 elements into tile.
    int i;
    for (i = 0; i < 32; i += blockDim.x) {
        if (i_n < ncols && (i_m + i) < nrows) {
            tile[threadIdx.y + i][threadIdx.x] = idata[(i_m + i) * ncols + i_n];
        }
    }
    __syncthreads();

    i_n = blockIdx.y * 32 + threadIdx.x;
    i_m = blockIdx.x * 32 + threadIdx.y;

    for (i = 0; i < 32; i += blockDim.x) {
        if (i_n < nrows && (i_m + i) < ncols) {
            odata[(i_m + i) * nrows + i_n] = tile[threadIdx.x][threadIdx.y + i];
        }
    }
}

//APPLICAR VECTOR GRADIENTE

__global__ void sumarAMatrizAMatrizB(float* a, float* b, int nrows, int ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        a[idx * ncols + idy] = a[idx * ncols + idy] + b[idx * ncols + idy];
    }
}

//A = A * B (cada elemento, no es un producto matricial); dim(A) = dim(B)
__global__ void multiplicarAMatrizAMatrizB(float* a, float* b, int nrows, int ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        a[idx * ncols + idy] = a[idx * ncols + idy] * b[idx * ncols + idy];
    }
}

//m-> [ [ m[i][j]/x for j = numero_columnas(m) ] for i = numero_filas(m) ]
__global__ void multiplicarCadaElementoMatriz(float* m, float x, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        m[idx * ncols + idy] = m[idx * ncols + idy] * x;
    }
}

/*__global__ void sumarACadaElementoVectorColumnaMatriz(float* m, float* v, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        atomicAdd(&v[idy], m[idx * ncols + idy]);
    }

}*/

__global__ void sumarACadaElementoVectorColumnaMatriz(const float* matrix, float* columnSums, int numRows, int numCols) {

    __shared__ float sharedMemory[32][32];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < numRows && idy < numCols) {
        sharedMemory[threadIdx.x][threadIdx.y] = matrix[idx * numCols + idy];
    }
    else {
        sharedMemory[threadIdx.x][threadIdx.y] = 0.0;
    }

    __syncthreads();

    int cnt = 2;
    for (int i = 32; i > 1; i = (int)(i / 2)) {
        if ((threadIdx.x + 1) % cnt == 0) {
            sharedMemory[threadIdx.x][threadIdx.y] += sharedMemory[threadIdx.x - (cnt / 2)][threadIdx.y];
        }
        cnt = cnt * 2;
        __syncthreads();
    }

    if (idy < numCols && threadIdx.x == 31) {
        atomicAdd(&columnSums[idy], sharedMemory[threadIdx.x][threadIdx.y]);
    }
    __syncthreads();

}

//con la matriz gradiente grad se aplica momento en la matriz destino mdst 
__global__ void actualizarValoresMatrizMomentoAdam(const float* grad, float* mdst, float b1, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        //if (mdst[idx * ncols + idy]!= 0) { printf("\nmomento distinto de 0"); }
        mdst[idx * ncols + idy] = b1* mdst[idx * ncols + idy] + (1-b1) * grad[idx * ncols + idy];
        //if (isnan(mdst[idx * ncols + idy]) || isinf(mdst[idx * ncols + idy])) { printf("\ncaca..."); }
    }
}

//con la matriz gradiente grad se aplica velocidad en la matriz destino mdst 
__global__ void actualizarValoresMatrizVelocidadAdam(const float* grad, float* mdst, float b2, int nrows, int ncols)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        //if (mdst[idx * ncols + idy] != 0) { printf("\nvelocidad distinto de 0"); }
        //if (mdst[idx * ncols + idy] < 0) { mdst[idx * ncols + idy] = -mdst[idx * ncols + idy]; }
        mdst[idx * ncols + idy] = b2 * mdst[idx * ncols + idy] + (1 - b2) * (grad[idx * ncols + idy] * grad[idx * ncols + idy]);
        //if (isnan(mdst[idx * ncols + idy]) || isinf(mdst[idx * ncols + idy])) { printf("\ncaca..."); }
    }
}

//se aplica el aprendizaje adam en la matriz destino mdest 
__global__ void calcularVectorGradienteAdam(float tapren, float b1, float b2, float epsilon, float* mdst, const float* mom, const float* vel, int nrows, int ncols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < nrows && idy < ncols) {
        float t1 = tapren * (mom[idx * ncols + idy] / (1 - b1));
        float t2 = epsilon + sqrtf(vel[idx * ncols + idy] / (float) (1 - b2));
        mdst[idx * ncols + idy] = -t1 / t2;
    }
}

__global__ void ponerTodosElementosVectorCero(float* v, int nelems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < nelems) {
        v[nelems] = 0.0;
        if (v[nelems] < 0) { 
            printf("\npero como puede ser posible tal tonteria??");
            v[nelems] = -v[nelems]; 
        }
    }
}