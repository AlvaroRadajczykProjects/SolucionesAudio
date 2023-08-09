#include "RedNeuronalSecuencial.cuh"

//para el producto de matrices utilizando cublas
cublasHandle_t handle;
const float alpha = 1.0f; //aparte del producto entre los elementos, puedes multiplicar esto
const float betamio = 0.0f; //aparte del producto final, puedes sumar esto

void imprimirVectorIntPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin) {
	printf("\n%s [ ", texto_mostrar);
	for (int i = inicio; i < fin; i++) {
		printf("%.8f", vector[i]);
		if (i < fin - 1) { printf(","); }
		printf(" ");
	}
	printf("]");
}

void imprimirMatrizPorPantalla(char* texto_mostrar, float matriz[], int n_filas, int n_columnas) {
	printf("\n%s\n", texto_mostrar);
	for (int i = 0; i < n_filas; i++) {
		imprimirVectorIntPorPantalla(" ", matriz, i * n_columnas, i * n_columnas + n_columnas);
	}
	printf("\n");
}

float vmax(float a, float b) {
	return a > b ? a : b;
}

dim3 dim3Ceil(float x, float y) {
	return dim3((int)ceil(x), (int)ceil(y));
}

const void aplicarFuncion(int id, float* zl, float* al, int nfilas, int ncolumnas) {
	if (id == 0) { aplicarFuncionSigmoideCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (zl, al, nfilas, ncolumnas); }
	else if (id == 1) { aplicarFuncionTahnCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (zl, al, nfilas, ncolumnas); }
	else if (id == 2) { aplicarFuncionCosenoEspecialCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (zl, al, nfilas, ncolumnas); }
	else if (id == 3) { aplicarFuncionPReluCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (zl, al, nfilas, ncolumnas); }
	//cudaDeviceSynchronize();
}

const void aplicarDerivadaFuncion(int id, float* m, int nfilas, int ncolumnas) {
	if (id == 0) { aplicarDerivadaFuncionSigmoideCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (m, nfilas, ncolumnas); }
	else if (id == 1) { aplicarDerivadaFuncionTahnCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (m, nfilas, ncolumnas); }
	else if (id == 2) { aplicarDerivadaFuncionCosenoEspecialCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (m, nfilas, ncolumnas); }
	else if (id == 3) { aplicarDerivadaFuncionPReluCadaElementoMatriz << < dim3Ceil(nfilas / (float)32, ncolumnas / (float)32), dim3(32, 32) >> > (m, nfilas, ncolumnas); }
	//cudaDeviceSynchronize();
}

const void productoMatricesDevice(const float* a, const float* b, float* c, int m, int k, int n) {
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, b, n, a, k, &betamio, c, n);
	//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, P, &alpha, d_B, P, d_A, N, &beta, d_C, P);
	/*productoMatrices << < dim3Ceil((p + 32 - 1) / (float)32, (m + 32 - 1) / (float)32), dim3(32, 32) >> > (a, b, c, m, n, p);
	cudaDeviceSynchronize();*/
}

//T(idata) = odata
const void matrizTraspuestaDevice(float* odata, float* idata, int m, int n) {
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &alpha, idata, n, &betamio, idata, m, odata, m);
	/*int dimension = (int)vmax(m, n);
	matrizTraspuesta << < dim3Ceil(dimension / (float)32, dimension / (float)32), dim3(32, 32) >> > (odata, idata, m, n);*/
	//cudaDeviceSynchronize();
}

void mostarMatrizDevice(char* com, float* p, int m, int n) {
	float* ph = new float[m * n];
	cudaMemcpy(ph, p, m * n * sizeof(float), cudaMemcpyDeviceToHost);
	imprimirMatrizPorPantalla(com, ph, m, n);
	delete ph;
}

//a*b = c; a, b, c, nf(a), nc(a) = nf(b), nc(b)
const void computeGold(const float* A, const float* B, float* C, unsigned int hA, unsigned int wA, unsigned int wB) {
	for (unsigned int i = 0; i < hA; ++i)
		for (unsigned int j = 0; j < wB; ++j) {
			float sum = 0;
			for (unsigned int k = 0; k < wA; ++k) {
				sum += A[i * wA + k] * B[k * wB + j];
			}
			C[i * wB + j] = sum;
		}
}

const void sumarVectores(const int tm, float* dst, const float* src) {
	for (int i = 0; i < tm; i++) {
		dst[i] += src[i];
	}
}

const void applyFastSigmoidFunction(const int tm, float* dst) {
	for (int i = 0; i < tm; i++) {
		dst[i] = (dst[i] / 1 + abs(dst[i])) * 0.5 + 0.5;
	}
}

//mejor que con una copia constante cambiar, que raro...
const void applyTahnFunction(const int tm, float* dst) {
	for (int i = 0; i < tm; i++) {
		dst[i] = tanh(dst[i]);
	}
}

const void applyPReluFunction(const int tm, float* dst) {
	for (int i = 0; i < tm; i++) {
		if (dst[i] < 0) { dst[i] = dst[i] * 0.01; }
		//else { dst[i] = dst[i]; }
	}
}

const void aplicarFuncionHost(int id, const int tm, float* dst) {
	if (id == 0) { applyFastSigmoidFunction(tm, dst); }
	else if (id == 1) { applyTahnFunction(tm, dst); }
	else if (id == 2) { applyPReluFunction(tm, dst); }
}

RedNeuronalSecuencial::RedNeuronalSecuencial(int nc, int* dc, int* fc) {
	numero_capas = nc;
	dimensiones_capas = dc;
	funciones_capas = fc;

	cargarEnDevice(true);
}

RedNeuronalSecuencial::RedNeuronalSecuencial(const char* nombre_archivo) {

	unsigned int nbytes = 0;

	char* cargar = leerArchivoYCerrar(nombre_archivo, &nbytes);

	unsigned int nnumeros = nbytes / 4;

	float* array = (float*)cargar;

	numero_capas = ((int*)array)[0];

	unsigned int offset = 1;

	dimensiones_capas = new int[numero_capas];
	funciones_capas = new int[numero_capas - 1];

	memcpy(dimensiones_capas, ((int*)array) + offset, numero_capas * sizeof(float));
	offset += numero_capas;

	memcpy(funciones_capas, ((int*)array) + offset, (numero_capas - 1) * sizeof(float));
	offset += numero_capas - 1;

	GestorPunteroPunteroFloatHost gestor_host_bias_vectors(numero_capas - 1, getCopiaDimensionesCapasRed());
	GestorPunteroPunteroFloatHost gestor_host_weight_matrices(numero_capas - 1, getCopiaDimensionesMatricesRed());

	float** host_bias_vectors = gestor_host_bias_vectors.getPunteroPunteroHost();
	float** host_weight_matrices = gestor_host_weight_matrices.getPunteroPunteroHost();

	for (int i = 1; i < numero_capas; i++) {
		memcpy(host_bias_vectors[i - 1], ((float*)array) + offset, dimensiones_capas[i] * sizeof(float));
		offset += dimensiones_capas[i];
	}

	for (int i = 0; i < numero_capas - 1; i++) {
		memcpy(host_weight_matrices[i], ((float*)array) + offset, dimensiones_capas[i] * dimensiones_capas[i + 1] * sizeof(float));
		offset += dimensiones_capas[i] * dimensiones_capas[i + 1];
	}

	cargarEnDevice(false);
	copiarPesosHostDevice(host_weight_matrices, host_bias_vectors);
}

void RedNeuronalSecuencial::cargarPunterosHostBiasesWeights(GestorPunteroPunteroFloatHost* biases, GestorPunteroPunteroFloatHost* weights) {

	biases = new GestorPunteroPunteroFloatHost(numero_capas - 1, getCopiaDimensionesCapasRed());
	weights = new GestorPunteroPunteroFloatHost(numero_capas - 1, getCopiaDimensionesMatricesRed());

	float** host_bias_vectors = biases->getPunteroPunteroHost();
	float** host_weight_matrices = weights->getPunteroPunteroHost();

	float** bias_vectors = device_bias_vectors->getPunteroPunteroHostDevice();
	float** weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

	for (int i = 1; i < numero_capas; i++) {
		cudaMemcpy(host_bias_vectors[i - 1], bias_vectors[i - 1], biases->getDimensionesElementos()[i - 1], cudaMemcpyDeviceToHost);
	}

	for (int i = 0; i < numero_capas - 1; i++) {
		cudaMemcpy(host_weight_matrices[i - 1], weight_matrices[i - 1], weights->getDimensionesElementos()[i - 1], cudaMemcpyDeviceToHost);
	}

	cudaDeviceSynchronize();

}

RedNeuronalSecuencial::~RedNeuronalSecuencial() {

	if (device_bias_vectors != NULL) { delete device_bias_vectors; device_bias_vectors = NULL; }
	if (device_weight_matrices != NULL) { delete device_weight_matrices; device_weight_matrices = NULL; }

	if (device_forward_zl != NULL) { delete device_forward_zl; device_forward_zl = NULL; }
	if (device_forward_al != NULL) { delete device_forward_al; device_forward_al = NULL; }

	if (device_err_weight_vgrad != NULL) { delete device_err_weight_vgrad; device_err_weight_vgrad = NULL; }
	if (device_err_bias_v != NULL) { delete device_err_bias_v; device_err_bias_v = NULL; }
	if (device_err_weight_v != NULL) { delete device_err_weight_v; device_err_weight_v = NULL; }

	if (device_batch_input != NULL) { cudaFree(device_batch_input); device_batch_input = NULL; }
	if (device_batch_output != NULL) { cudaFree(device_batch_output); device_batch_output = NULL; }
	if (temp_matr_traspose != NULL) { cudaFree(temp_matr_traspose); temp_matr_traspose = NULL; }

	if (dimensiones_capas != NULL) { free(dimensiones_capas); dimensiones_capas = NULL; }
	if (funciones_capas != NULL) { free(funciones_capas); funciones_capas = NULL; }

	cudaDeviceSynchronize();

}

int RedNeuronalSecuencial::getNumeroCapas() {
	return numero_capas;
}

int* RedNeuronalSecuencial::getDimensionesCapas() {
	return dimensiones_capas;
}

int* RedNeuronalSecuencial::getFuncionesCapas() {
	return funciones_capas;
}

int* RedNeuronalSecuencial::getCopiaDimensionesCapasRed() {
	int* copia_dimensiones_capas = new int[numero_capas - 1];
	for (int i = 0; i < numero_capas - 1; i++) { copia_dimensiones_capas[i] = dimensiones_capas[i + 1]; }
	return copia_dimensiones_capas;
}

int* RedNeuronalSecuencial::getCopiaDimensionesMatricesRed() {
	int* dimensiones_matrices = new int[numero_capas - 1];
	for (int i = 0; i < numero_capas - 1; i++) { dimensiones_matrices[i] = dimensiones_capas[i] * dimensiones_capas[i + 1]; }
	return dimensiones_matrices;
}

void RedNeuronalSecuencial::exportarRedComoArchivo(const char* nombre_archivo) {

	GestorPunteroPunteroFloatHost gestor_host_bias_vectors(numero_capas - 1, getCopiaDimensionesCapasRed());
	GestorPunteroPunteroFloatHost gestor_host_weight_matrices(numero_capas - 1, getCopiaDimensionesMatricesRed());

	float** host_bias_vectors = gestor_host_bias_vectors.getPunteroPunteroHost();
	float** host_weight_matrices = gestor_host_weight_matrices.getPunteroPunteroHost();

	device_bias_vectors->copiarDeviceAHost(host_bias_vectors);
	device_weight_matrices->copiarDeviceAHost(host_weight_matrices);

	unsigned int numero = 1 + numero_capas + (numero_capas - 1);
	for (int i = 1; i < numero_capas; i++) { numero += dimensiones_capas[i]; }
	for (int i = 1; i < numero_capas; i++) { numero += dimensiones_capas[i] * dimensiones_capas[i - 1]; }

	float* array = (float*)malloc(numero * sizeof(float));
	((int*)array)[0] = numero_capas;

	unsigned int offset = 1;

	memcpy(((int*)array) + offset, dimensiones_capas, numero_capas * sizeof(float));
	offset += numero_capas;

	memcpy(((int*)array) + offset, funciones_capas, (numero_capas - 1) * sizeof(float));
	offset += numero_capas - 1;

	for (int i = 1; i < numero_capas; i++) {
		memcpy(((float*)array) + offset, host_bias_vectors[i - 1], dimensiones_capas[i] * sizeof(float));
		offset += dimensiones_capas[i];
	}

	for (int i = 0; i < numero_capas - 1; i++) {
		memcpy(((float*)array) + offset, host_weight_matrices[i], dimensiones_capas[i] * dimensiones_capas[i + 1] * sizeof(float));
		offset += dimensiones_capas[i] * dimensiones_capas[i + 1];
	}

	char* buffer = (char*)array;

	crearArchivoEscribirYCerrar(nombre_archivo, numero * sizeof(float), buffer);

	cudaFree(array);
	array = NULL;

	//se limpian solos los punteros, ojo también, no limpies host_bias_vectors ni host_weight_matrices en el destructor de esta clase
}

void RedNeuronalSecuencial::copiarPesosHostDevice(float** host_pesos, float** host_biases) {
	device_bias_vectors->copiarHostADevice(host_biases);
	device_weight_matrices->copiarHostADevice(host_pesos);
	cudaDeviceSynchronize();
}

void RedNeuronalSecuencial::cargarEnDevice(bool iniciarValoresBiasesWeights) {

	device_bias_vectors = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesCapasRed());
	device_weight_matrices = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesMatricesRed());

	float** puntero_host_device = device_bias_vectors->getPunteroPunteroHostDevice();
	float** punteros_en_host_de_device_weights = device_weight_matrices->getPunteroPunteroHostDevice();

	if (iniciarValoresBiasesWeights) {
		//de esta manera se ponen todos los valores de los biases a 0's
		for (int i = 0; i < device_bias_vectors->getNumeroElementos(); i++) {
			cudaMemset(puntero_host_device[i], 0, device_bias_vectors->getDimensionesElementos()[i] * sizeof(float));
		}

		//ahora estableceremos los valores aleatorios de los pesos
		curandGenerator_t generador_dnorm = crearGeneradorNumerosAleatoriosEnDistribucionNormal();

		for (int i = 0; i < device_weight_matrices->getNumeroElementos(); i++) {
			//generarNumerosAleatoriosEnDistribucionNormal(generador_dnorm, 0.0, 1.0, punteros_en_host_de_device_weights[i], device_weight_matrices->getDimensionesElementos()[i]);
			//min(1.0,2/(float)device_bias_vectors->getDimensionesElementos()[i])
			generarNumerosAleatoriosEnDistribucionNormal(generador_dnorm, 0.0, 2 / (float)device_bias_vectors->getDimensionesElementos()[i], punteros_en_host_de_device_weights[i], device_weight_matrices->getDimensionesElementos()[i]);
		}

		curandDestroyGenerator(generador_dnorm);
	}

	cudaDeviceSynchronize();

}

float* RedNeuronalSecuencial::propagacionHaciaDelante(int nejemplos, int nvalsentrada, float* matrizejemplos) {

	if (nvalsentrada == dimensiones_capas[0]) {

		int dimension_host = dimensiones_capas[numero_capas - 1];
		int dimension_device = 0;
		for (int i = 0; i < numero_capas; i++) { dimension_device = (int)vmax(dimension_device, dimensiones_capas[i]); }

		float* host_matriz_resultado = (float*)malloc(nejemplos * dimension_host * sizeof(float));

		float* device_matriz_entrada = 0;
		cudaMalloc(&device_matriz_entrada, nejemplos * dimension_device * sizeof(float));
		cudaMemcpy(device_matriz_entrada, matrizejemplos, nejemplos * nvalsentrada * sizeof(float), cudaMemcpyHostToDevice);

		float* device_matriz_resultado = 0;
		cudaMalloc(&device_matriz_resultado, nejemplos * dimension_device * sizeof(float));

		float** host_device_bias_vectors = device_bias_vectors->getPunteroPunteroHostDevice();
		float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

		int M = nejemplos;
		int N = 0;
		int P = 0;

		for (int i = 0; i < numero_capas - 1; i++) {
			N = dimensiones_capas[i];
			P = dimensiones_capas[i + 1];

			productoMatricesDevice(device_matriz_entrada, host_device_weight_matrices[i], device_matriz_resultado, M, N, P);

			sumarCadaFilaMatrizVector << < dim3Ceil(M / (float)32, P / (float)32), dim3(32, 32) >> > (device_matriz_resultado, host_device_bias_vectors[i], M, P);
			//cudaDeviceSynchronize();

			aplicarFuncion(funciones_capas[i], device_matriz_resultado, device_matriz_resultado, M, P);
			//aplicarFuncionSigmoideCadaElementoMatriz << < dim3Ceil(M / (float)32, P / (float)32), dim3(32, 32) >> > (device_matriz_resultado, device_matriz_resultado, M , P);
			//manageCUDAError(cudaDeviceSynchronize());

			cudaMemcpy(device_matriz_entrada, device_matriz_resultado, M * P * sizeof(float), cudaMemcpyDeviceToDevice);
		}

		cudaMemcpy(host_matriz_resultado, device_matriz_resultado, nejemplos * dimension_host * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(device_matriz_resultado);
		device_matriz_resultado = NULL;

		return host_matriz_resultado;
	}
	else {
		printf("El numero de valores de cada ejemplo debe ser igual que el tamanno de la capa de entrada de la red");
		return NULL;
	}
}

int RedNeuronalSecuencial::getMaxTamMatrTempTrans(int batch_size) {
	int* dimensiones_matrices = getCopiaDimensionesMatricesRed();
	int mayor_dimension_matriz = 0;
	for (int i = 0; i < numero_capas - 1; i++) { mayor_dimension_matriz = (int)vmax(mayor_dimension_matriz, dimensiones_matrices[i]); }
	free(dimensiones_matrices);
	dimensiones_matrices = NULL;

	int mayor_tam_capa = 0;
	for (int i = 0; i < numero_capas; i++) { mayor_tam_capa = (int)vmax(mayor_tam_capa, dimensiones_capas[i]); }
	return ((int)vmax(mayor_dimension_matriz, batch_size * mayor_tam_capa));
}

int* RedNeuronalSecuencial::getDimensionesZlAl(int batch_size) {
	int* dimensionesCapas = getCopiaDimensionesCapasRed();
	for (int i = 0; i < numero_capas - 1; i++) { dimensionesCapas[i] = dimensionesCapas[i] * batch_size; }
	return dimensionesCapas;
}

float RedNeuronalSecuencial::calcularFuncionCosteMSE(int batch_size, int nvalsalida, float* vsalida) {

	if (nvalsalida == dimensiones_capas[numero_capas - 1]) {

		float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

		int os = batch_size * nvalsalida * sizeof(float);
		float* copia_vsalida = 0;
		cudaMalloc(&copia_vsalida, os);

		cudaMemcpy(copia_vsalida, vsalida, os, cudaMemcpyHostToDevice);

		aplicarFuncionCosteMSE << < dim3Ceil(batch_size / (float)32, nvalsalida / (float)32), dim3(32, 32) >> > (batch_size, nvalsalida, host_device_al[numero_capas - 2], copia_vsalida, copia_vsalida);
		//cudaDeviceSynchronize();

		float* agrupados = 0;
		cudaMalloc(&agrupados, nvalsalida * sizeof(float));

		cudaMemset(agrupados, 0, nvalsalida * sizeof(float));

		sumarACadaElementoVectorColumnaMatriz << < dim3Ceil(batch_size / (float)32, nvalsalida / (float)32), dim3(32, 32) >> > (copia_vsalida, agrupados, batch_size, nvalsalida);

		float* resultado = (float*)malloc(nvalsalida * sizeof(float));
		cudaMemcpy(resultado, agrupados, nvalsalida * sizeof(float), cudaMemcpyDeviceToHost);

		cudaFree(copia_vsalida);
		cudaFree(agrupados);

		float res = 0.0;
		for (int i = 0; i < nvalsalida; i++) { res += resultado[i]; }

		free(resultado);

		//printf("\nValor de la funcion de coste MSE: %.16f",res/(batch_size*nvalsalida));

		return res;

	}

	return 0.0;
}



/* --------------------------------------------------------------- ENTRENAMIENTO SGD --------------------------------------------------------------- */



void RedNeuronalSecuencial::entrenarRedMSE_SGD(float tapren, int mostrar_fcoste_cada_n_epocas, int nepocas, int nejemplos, int batch_size, int nvalsentrada, int nvalsalida, float* ventrada, float* vsalida) {

	int ins = batch_size * nvalsentrada * sizeof(float);
	int os = batch_size * nvalsalida * sizeof(float);

	cublasCreate(&handle);

	if (nvalsentrada == dimensiones_capas[0] && nvalsalida == dimensiones_capas[numero_capas - 1]) {

		cudaMalloc(&device_batch_input, ins);
		cudaMalloc(&device_batch_output, os);
		cudaMalloc(&temp_matr_traspose, getMaxTamMatrTempTrans(batch_size) * sizeof(float));

		device_forward_zl = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));
		device_forward_al = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));

		device_err_bias_vgrad = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesCapasRed());
		device_err_weight_vgrad = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesMatricesRed());

		for (int i = 0; i < nepocas; i++) {
			float error = 0.0;
			int nbatchs = (int)(nejemplos / batch_size);
			int nrelems = nejemplos % batch_size;
			for (int j = 0; j < nbatchs; j++) {
				cudaMemcpy(device_batch_input, ventrada + (batch_size * nvalsentrada * j), ins, cudaMemcpyHostToDevice);
				cudaMemcpy(device_batch_output, vsalida + (batch_size * nvalsalida * j), os, cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada + (batch_size * nvalsentrada * j));
				error += calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida + (batch_size * nvalsalida * j));
				calcularVectorGradiente(batch_size, nvalsalida, vsalida + (batch_size * nvalsalida * j));
				aplicarVectorGradienteSGD(tapren, batch_size);
			}
			//aquí se hace con el resto
			if (nrelems > 0) {
				cudaMemcpy(device_batch_input, ventrada + (batch_size * nvalsentrada * nbatchs), nrelems * nvalsentrada * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(device_batch_output, vsalida + (batch_size * nvalsalida * nbatchs), nrelems * nvalsalida * sizeof(float), cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				propagacionHaciaDelanteEntrenamiento(nrelems, nvalsentrada, ventrada + (batch_size * nvalsentrada * nbatchs));
				error += calcularFuncionCosteMSE(nrelems, nvalsalida, vsalida + (batch_size * nvalsalida * nbatchs));
				calcularVectorGradiente(nrelems, nvalsalida, vsalida + (batch_size * nvalsalida * nbatchs));
				aplicarVectorGradienteSGD(tapren, nrelems);
			}
			if ((i + 1) % mostrar_fcoste_cada_n_epocas == 0) {
				printf("\nError MSE: %.16f | Quedan %d epocas", (float)(error / ((float)(nejemplos * nvalsalida))), nepocas - i - 1);
			}
			if (error < 0.000000000001) {
				printf("\nSe ha convergido bastante bien");
				break;
			}
		}

		/*
		cudaMemcpy(device_batch_input, ventrada, ins, cudaMemcpyHostToDevice);
		cudaMemcpy(device_batch_output, vsalida, os, cudaMemcpyHostToDevice);


		for (int i = 0; i < nepocas; i++) {
			float error = 0.0;
			propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada);
			if ((i + 1) % 500 == 0) {
				error += calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida);
				printf("\nError MSE: %.16f | Quedan %d epocas", error/(float)(nejemplos*nvalsalida), nepocas - i - 1);
			}
			calcularVectorGradiente(batch_size, nvalsalida, vsalida);
			aplicarVectorGradienteSGD(tapren, batch_size);
		}
		printf("\n\nValor final de la funcion de coste:");
		propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada);
		calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida);
		*/

		if (device_batch_input != NULL) { cudaFree(device_batch_input); device_batch_input = NULL; }
		if (device_batch_output != NULL) { cudaFree(device_batch_output); device_batch_output = NULL; }
		if (temp_matr_traspose != NULL) { cudaFree(temp_matr_traspose); temp_matr_traspose = NULL; }

		if (device_forward_zl != NULL) { delete device_forward_zl; device_forward_zl = NULL; }
		if (device_forward_al != NULL) { delete device_forward_al; device_forward_al = NULL; }

		if (device_err_bias_vgrad != NULL) { delete device_err_bias_vgrad; device_err_bias_vgrad = NULL; }
		if (device_err_weight_vgrad != NULL) { delete device_err_weight_vgrad; device_err_weight_vgrad = NULL; }

		cublasDestroy(handle);

	}
	else {
		printf("El numero de valores de cada ejemplo de entrada y salida debe ser igual que el tamanno de la capa de entrada de la red");
	}

}

void RedNeuronalSecuencial::calcularVectorGradiente(int batch_size, int nvalsalida, float* vsalida) {

	float** host_device_zl = device_forward_zl->getPunteroPunteroHostDevice();
	float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

	float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

	float** host_device_weight_error_matrices = device_err_weight_vgrad->getPunteroPunteroHostDevice();

	aplicarDerivadaFuncionPerdidaMSECadaElementoPredY << < dim3Ceil(batch_size / (float)32, nvalsalida / (float)32), dim3(32, 32) >> > (batch_size, nvalsalida, host_device_al[numero_capas - 2], device_batch_output);
	//cudaDeviceSynchronize();

	for (int i = numero_capas - 1; i > 0; i--) {

		//error bias actual

		aplicarDerivadaFuncion(funciones_capas[i - 1], host_device_zl[i - 1], batch_size, dimensiones_capas[i]);
		//aplicarDerivadaFuncionSigmoideCadaElementoMatriz << < dim3Ceil(batch_size / (float)32, dimensiones_capas[i] / (float)32), dim3(32, 32) >> > (host_device_zl[i - 1], batch_size, dimensiones_capas[i]);
		//manageCUDAError(cudaDeviceSynchronize());

		multiplicarAMatrizAMatrizB << < dim3Ceil(batch_size / (float)32, dimensiones_capas[i] / (float)32), dim3(32, 32) >> > (host_device_al[i - 1], host_device_zl[i - 1], batch_size, dimensiones_capas[i]);
		//cudaDeviceSynchronize();

		//error pesos

		if (i > 1) {
			matrizTraspuestaDevice(temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i - 1]);
		}
		else {
			matrizTraspuestaDevice(temp_matr_traspose, device_batch_input, batch_size, dimensiones_capas[i - 1]);
		}

		productoMatricesDevice(temp_matr_traspose, host_device_al[i - 1], host_device_weight_error_matrices[i - 1], dimensiones_capas[i - 1], batch_size, dimensiones_capas[i]);

		//error bias anterior

		if (i > 1) {
			matrizTraspuestaDevice(temp_matr_traspose, host_device_weight_matrices[i - 1], dimensiones_capas[i - 1], dimensiones_capas[i]);

			productoMatricesDevice(host_device_al[i - 1], temp_matr_traspose, host_device_al[i - 2], batch_size, dimensiones_capas[i], dimensiones_capas[i - 1]);
		}

	}
}

void RedNeuronalSecuencial::aplicarVectorGradienteSGD(float tapren, int batch_size) {

	float factor = -1.0 * (tapren / (float) batch_size);

	float** host_device_bias_vectors = device_bias_vectors->getPunteroPunteroHostDevice();
	float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

	float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

	float** host_device_bias_error_matrices = device_err_bias_vgrad->getPunteroPunteroHostDevice();
	float** host_device_weight_error_matrices = device_err_weight_vgrad->getPunteroPunteroHostDevice();

	for (int i = 0; i < numero_capas - 1; i++) {

		//aquí se hacen los biases

		dim3 dimension = dim3Ceil(batch_size / (float)32, dimensiones_capas[i + 1] / (float)32);

		sumarACadaElementoVectorColumnaMatriz << < dimension, dim3(32, 32) >> > (host_device_al[i], host_device_bias_error_matrices[i], batch_size, dimensiones_capas[i + 1]);

		int nbloques = ( (int) ceil(dimensiones_capas[i + 1] / (float)1024) );

		multiplicarCadaElementoMatriz << < nbloques, 1024 >> > (host_device_bias_error_matrices[i], factor, 1, dimensiones_capas[i + 1]);

		sumarAMatrizAMatrizB << < nbloques, 1024 >> > (host_device_bias_vectors[i], host_device_bias_error_matrices[i], 1, dimensiones_capas[i + 1]);

		//aquí se hacen los pesos

		dimension = dim3Ceil(dimensiones_capas[i] / (float)32, dimensiones_capas[i + 1] / (float)32);

		multiplicarCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_weight_error_matrices[i], factor, dimensiones_capas[i], dimensiones_capas[i + 1]);

		sumarAMatrizAMatrizB << < dimension, dim3(32, 32) >> > (host_device_weight_matrices[i], host_device_weight_error_matrices[i], dimensiones_capas[i], dimensiones_capas[i + 1]);
	}
}



/* --------------------------------------------------------------- ENTRENAMIENTO ADAM --------------------------------------------------------------- */



void RedNeuronalSecuencial::entrenarRedMSE_Adam(float tapren, float b1, float b2, float epsilon, int mostrar_fcoste_cada_n_epocas, int nepocas, int nejemplos, int batch_size, int nvalsentrada, int nvalsalida, float* ventrada, float* vsalida) {

	int ins = batch_size * nvalsentrada * sizeof(float);
	int os = batch_size * nvalsalida * sizeof(float);

	cublasCreate(&handle);

	if (nvalsentrada == dimensiones_capas[0] && nvalsalida == dimensiones_capas[numero_capas - 1]) {

		cudaMalloc(&device_batch_input, ins);
		cudaMalloc(&device_batch_output, os);
		cudaMalloc(&temp_matr_traspose, getMaxTamMatrTempTrans(batch_size) * sizeof(float));

		device_forward_zl = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));
		device_forward_al = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getDimensionesZlAl(batch_size));

		device_err_bias_vgrad = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesCapasRed());
		device_err_weight_vgrad = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesMatricesRed());

		device_err_bias_m = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesCapasRed());
		device_err_weight_m = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesMatricesRed());
		device_err_bias_v = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesCapasRed());
		device_err_weight_v = new GestorPunteroPunteroFloatDevice(numero_capas - 1, getCopiaDimensionesMatricesRed());

		device_err_bias_m->ponerElementosTodosElementosACero();
		device_err_weight_m->ponerElementosTodosElementosACero();
		device_err_bias_v->ponerElementosTodosElementosACero();
		device_err_weight_v->ponerElementosTodosElementosACero();

		cudaDeviceSynchronize();

		/*GestorPunteroPunteroFloatHost p(numero_capas - 1, getDimensionesZlAl(batch_size));
		GestorPunteroPunteroFloatHost p2(numero_capas - 1, getCopiaDimensionesMatricesRed());
		GestorPunteroPunteroFloatHost p3(numero_capas - 1, getDimensionesZlAl(batch_size));
		GestorPunteroPunteroFloatHost p4(numero_capas - 1, getCopiaDimensionesMatricesRed());
		device_err_bias_m->copiarDeviceAHost(p.getPunteroPunteroHost());
		device_err_weight_m->copiarDeviceAHost(p2.getPunteroPunteroHost());
		device_err_bias_v->copiarDeviceAHost(p3.getPunteroPunteroHost());
		device_err_weight_v->copiarDeviceAHost(p4.getPunteroPunteroHost());

		for (int i = 0; i < numero_capas - 1; i++) {
			printf("\ncapa %d\n",i);
			imprimirMatrizPorPantalla("", p.getPunteroPunteroHost()[i], batch_size, dimensiones_capas[i+1]);
			imprimirMatrizPorPantalla("", p2.getPunteroPunteroHost()[i], dimensiones_capas[i], dimensiones_capas[i + 1]);
			imprimirMatrizPorPantalla("", p3.getPunteroPunteroHost()[i], batch_size, dimensiones_capas[i + 1]);
			imprimirMatrizPorPantalla("", p4.getPunteroPunteroHost()[i], dimensiones_capas[i], dimensiones_capas[i + 1]);
		}*/

		for (int i = 0; i < nepocas; i++) {
			float error = 0.0;
			int nbatchs = (int)(nejemplos / batch_size);
			int nrelems = nejemplos % batch_size;
			for (int j = 0; j < nbatchs; j++) {
				cudaMemcpy(device_batch_input, ventrada + (batch_size * nvalsentrada * j), ins, cudaMemcpyHostToDevice);
				cudaMemcpy(device_batch_output, vsalida + (batch_size * nvalsalida * j), os, cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada + (batch_size * nvalsentrada * j));
				error += calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida + (batch_size * nvalsalida * j));	
				calcularVectorGradiente(batch_size, nvalsalida, vsalida + (batch_size * nvalsalida * j));
				aplicarVectorGradienteAdam(tapren, b1, b2, epsilon, batch_size);
			}
			//aquí se hace con el resto
			if (nrelems > 0) {
				cudaMemcpy(device_batch_input, ventrada + (batch_size * nvalsentrada * nbatchs), nrelems * nvalsentrada * sizeof(float), cudaMemcpyHostToDevice);
				cudaMemcpy(device_batch_output, vsalida + (batch_size * nvalsalida * nbatchs), nrelems * nvalsalida * sizeof(float), cudaMemcpyHostToDevice);
				cudaDeviceSynchronize();
				propagacionHaciaDelanteEntrenamiento(nrelems, nvalsentrada, ventrada + (batch_size * nvalsentrada * nbatchs));
				error += calcularFuncionCosteMSE(nrelems, nvalsalida, vsalida + (batch_size * nvalsalida * nbatchs));
				calcularVectorGradiente(nrelems, nvalsalida, vsalida + (batch_size * nvalsalida * nbatchs));
				aplicarVectorGradienteAdam(tapren, b1, b2, epsilon, nrelems);
			}
			if ((i + 1) % mostrar_fcoste_cada_n_epocas == 0) {
				printf("\nError MSE: %.16f | Quedan %d epocas", (float)(error / ((float)(nejemplos * nvalsalida))), nepocas - i - 1);
			}
			if (error < 0.000000000001) { 
				printf("\nSe ha convergido bastante bien");
				break; 
			}
		}

		/*
		cudaMemcpy(device_batch_input, ventrada, ins, cudaMemcpyHostToDevice);
		cudaMemcpy(device_batch_output, vsalida, os, cudaMemcpyHostToDevice);


		for (int i = 0; i < nepocas; i++) {
			float error = 0.0;
			propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada);
			if ((i + 1) % 500 == 0) {
				error += calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida);
				printf("\nError MSE: %.16f | Quedan %d epocas", error/(float)(nejemplos*nvalsalida), nepocas - i - 1);
			}
			calcularVectorGradiente(batch_size, nvalsalida, vsalida);
			aplicarVectorGradienteSGD(tapren, batch_size);
		}
		printf("\n\nValor final de la funcion de coste:");
		propagacionHaciaDelanteEntrenamiento(batch_size, nvalsentrada, ventrada);
		calcularFuncionCosteMSE(batch_size, nvalsalida, vsalida);
		*/

		if (device_batch_input != NULL) { cudaFree(device_batch_input); device_batch_input = NULL; }
		if (device_batch_output != NULL) { cudaFree(device_batch_output); device_batch_output = NULL; }
		if (temp_matr_traspose != NULL) { cudaFree(temp_matr_traspose); temp_matr_traspose = NULL; }

		if (device_forward_zl != NULL) { delete device_forward_zl; device_forward_zl = NULL; }
		if (device_forward_al != NULL) { delete device_forward_al; device_forward_al = NULL; }

		if (device_err_bias_vgrad != NULL) { delete device_err_bias_vgrad; device_err_bias_vgrad = NULL; }
		if (device_err_weight_vgrad != NULL) { delete device_err_weight_vgrad; device_err_weight_vgrad = NULL; }

		if (device_err_bias_m != NULL) { delete device_err_bias_m; device_err_bias_m = NULL; }
		if (device_err_weight_m != NULL) { delete device_err_weight_m; device_err_weight_m = NULL; }
		if (device_err_bias_v != NULL) { delete device_err_bias_v; device_err_bias_v = NULL; }
		if (device_err_weight_v != NULL) { delete device_err_weight_v; device_err_weight_v = NULL; }

		cublasDestroy(handle);

	}
	else {
		printf("El numero de valores de cada ejemplo de entrada y salida debe ser igual que el tamanno de la capa de entrada de la red");
	}

}

void RedNeuronalSecuencial::aplicarVectorGradienteAdam(float tapren, float b1, float b2, float epsilon, int batch_size) {
	float factor = 1.0 / (float) batch_size;

	float** host_device_bias_vectors = device_bias_vectors->getPunteroPunteroHostDevice();
	float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

	float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

	float** host_device_bias_error_matrices = device_err_bias_vgrad->getPunteroPunteroHostDevice();
	float** host_device_weight_error_matrices = device_err_weight_vgrad->getPunteroPunteroHostDevice();

	float** host_device_bias_m = device_err_bias_m->getPunteroPunteroHostDevice();
	float** host_device_weight_m = device_err_weight_m->getPunteroPunteroHostDevice();
	float** host_device_bias_v = device_err_bias_v->getPunteroPunteroHostDevice();
	float** host_device_weight_v = device_err_weight_v->getPunteroPunteroHostDevice();

	for (int i = 0; i < numero_capas - 1; i++) {

		//aquí se hacen los biases

		dim3 dimension = dim3Ceil(batch_size / (float)32, dimensiones_capas[i + 1] / (float)32);

		sumarACadaElementoVectorColumnaMatriz << < dimension, dim3(32, 32) >> > (host_device_al[i], host_device_bias_error_matrices[i], batch_size, dimensiones_capas[i + 1]);

		int nbloques = ((int)ceil(dimensiones_capas[i + 1] / (float)1024));

		multiplicarCadaElementoMatriz << < nbloques, 1024 >> > (host_device_bias_error_matrices[i], factor, 1, dimensiones_capas[i + 1]);

		actualizarValoresMatrizMomentoAdam << < nbloques, 1024 >> > (host_device_bias_error_matrices[i], host_device_bias_m[i], b1, 1, dimensiones_capas[i + 1]);
		actualizarValoresMatrizVelocidadAdam << < nbloques, 1024 >> > (host_device_bias_error_matrices[i], host_device_bias_v[i], b2, 1, dimensiones_capas[i + 1]);

		calcularVectorGradienteAdam << < nbloques, 1024 >> > (tapren, b1, b2, epsilon, host_device_bias_error_matrices[i], host_device_bias_m[i], host_device_bias_v[i], 1, dimensiones_capas[i + 1]);

		sumarAMatrizAMatrizB << < nbloques, 1024 >> > (host_device_bias_vectors[i], host_device_bias_error_matrices[i], 1, dimensiones_capas[i + 1]);

		//aquí se hacen los pesos

		dimension = dim3Ceil(dimensiones_capas[i] / (float)32, dimensiones_capas[i + 1] / (float)32);

		multiplicarCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_weight_error_matrices[i], factor, dimensiones_capas[i], dimensiones_capas[i + 1]);

		actualizarValoresMatrizMomentoAdam << < dimension, dim3(32, 32) >> > (host_device_weight_error_matrices[i], host_device_weight_m[i], b1, dimensiones_capas[i], dimensiones_capas[i + 1]);
		actualizarValoresMatrizVelocidadAdam << < dimension, dim3(32, 32) >> > (host_device_weight_error_matrices[i], host_device_weight_v[i], b2, dimensiones_capas[i], dimensiones_capas[i + 1]);

		calcularVectorGradienteAdam << < dimension, dim3(32, 32) >> > (tapren, b1, b2, epsilon, host_device_weight_error_matrices[i], host_device_weight_m[i], host_device_weight_v[i], dimensiones_capas[i], dimensiones_capas[i + 1]);

		sumarAMatrizAMatrizB << < dimension, dim3(32, 32) >> > (host_device_weight_matrices[i], host_device_weight_error_matrices[i], dimensiones_capas[i], dimensiones_capas[i + 1]);
	}

	/*
	for (int i = 0; i < numero_capas - 1; i++) {

		dim3 dimension = dim3Ceil(batch_size / (float)32, dimensiones_capas[i + 1] / (float)32);

		sumarACadaElementoVectorColumnaMatriz << < dimension, dim3(32, 32) >> > (host_device_al[i], host_device_bias_error_matrices[i], batch_size, dimensiones_capas[i + 1]);
		
		dimension = dim3Ceil(1, dimensiones_capas[i + 1] / (float)1024);

		cudaMemcpy(host_device_bias_v[i], host_device_bias_m[i], dimensiones_capas[i + 1]*sizeof(float), cudaMemcpyDeviceToDevice);
		//sumarACadaElementoVectorColumnaMatriz << < dimension, dim3(32, 32) >> > (host_device_al[i], host_device_bias_v[i], batch_size, dimensiones_capas[i + 1]);

		multiplicarCadaElementoMatriz << < dimension, dim3(1, 1024) >> > (host_device_bias_error_matrices[i], factor, 1, dimensiones_capas[i + 1]);

		actualizarValoresMatrizMomentoAdam << < dimension, dim3(1, 1024) >> > (host_device_bias_error_matrices[i], host_device_bias_m[i], b1, 1, dimensiones_capas[i + 1]);
		actualizarValoresMatrizVelocidadAdam << < dimension, dim3(1, 1024) >> > (host_device_bias_error_matrices[i], host_device_bias_v[i], b2, 1, dimensiones_capas[i + 1]);

		calcularVectorGradienteAdam <<< dimension, dim3(1, 1024) >>>  (tapren, b1, b2, epsilon, host_device_bias_error_matrices[i], host_device_bias_m[i], host_device_bias_v[i], 1, dimensiones_capas[i + 1]);
		
		sumarAMatrizAMatrizB << < dimension, dim3(1, 1024) >> > (host_device_bias_vectors[i], host_device_bias_error_matrices[i], 1, dimensiones_capas[i + 1]);

		//sumarACadaElementoVectorColumnaMatriz << < dimension, dim3(32, 32) >> > (host_device_al[i], host_device_bias_vectors[i], batch_size, dimensiones_capas[i + 1]);
		//cudaDeviceSynchronize();

		dimension = dim3Ceil(dimensiones_capas[i] / (float)32, dimensiones_capas[i + 1] / (float)32);

		multiplicarCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_weight_error_matrices[i], factor, dimensiones_capas[i], dimensiones_capas[i + 1]);

		actualizarValoresMatrizMomentoAdam << < dimension, dim3(32, 32) >> > (host_device_weight_error_matrices[i], host_device_weight_m[i], b1, dimensiones_capas[i], dimensiones_capas[i + 1]);
		actualizarValoresMatrizVelocidadAdam << < dimension, dim3(32, 32) >> > (host_device_weight_error_matrices[i], host_device_weight_v[i], b2, dimensiones_capas[i], dimensiones_capas[i + 1]);

		calcularVectorGradienteAdam << < dimension, dim3(32, 32) >> > (tapren, b1, b2, epsilon, host_device_weight_error_matrices[i], host_device_weight_m[i], host_device_weight_v[i], dimensiones_capas[i], dimensiones_capas[i + 1]);

		sumarAMatrizAMatrizB << < dimension, dim3(32, 32) >> > (host_device_weight_matrices[i], host_device_weight_error_matrices[i], dimensiones_capas[i], dimensiones_capas[i + 1]);
		//cudaDeviceSynchronize();
	}
	*/
}

void RedNeuronalSecuencial::propagacionHaciaDelanteEntrenamiento(int nejemplos, int nvalsentrada, float* matrizejemplos) {

	float** host_device_bias_vectors = device_bias_vectors->getPunteroPunteroHostDevice();
	float** host_device_weight_matrices = device_weight_matrices->getPunteroPunteroHostDevice();

	float** host_device_zl = device_forward_zl->getPunteroPunteroHostDevice();
	float** host_device_al = device_forward_al->getPunteroPunteroHostDevice();

	int M = nejemplos;
	int N = dimensiones_capas[0];
	int P = dimensiones_capas[1];

	productoMatricesDevice(device_batch_input, host_device_weight_matrices[0], host_device_zl[0], M, N, P);

	dim3 dimension = dim3Ceil(M / (float)32, P / (float)32);
	sumarCadaFilaMatrizVector << < dimension, dim3(32, 32) >> > (host_device_zl[0], host_device_bias_vectors[0], M, P);
	//cudaDeviceSynchronize();

	aplicarFuncion(funciones_capas[0], host_device_zl[0], host_device_al[0], M, P);
	//aplicarFuncionSigmoideCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_zl[0], host_device_al[0], M, P);
	//manageCUDAError(cudaDeviceSynchronize());

	for (int i = 1; i < numero_capas - 1; i++) {
		N = dimensiones_capas[i];
		P = dimensiones_capas[i + 1];
		productoMatricesDevice(host_device_al[i - 1], host_device_weight_matrices[i], host_device_zl[i], M, N, P);

		dim3 dimension = dim3Ceil(M / (float)32, P / (float)32);

		sumarCadaFilaMatrizVector << < dimension, dim3(32, 32) >> > (host_device_zl[i], host_device_bias_vectors[i], M, P);
		//cudaDeviceSynchronize();

		aplicarFuncion(funciones_capas[i], host_device_zl[i], host_device_al[i], M, P);
		//aplicarFuncionSigmoideCadaElementoMatriz << < dimension, dim3(32, 32) >> > (host_device_zl[i], host_device_al[i], M, P);
		//manageCUDAError(cudaDeviceSynchronize());
	}

}

void RedNeuronalSecuencial::iniciarModoPropagacionDelanteRapido() {
	host_bias_vectors = new GestorPunteroPunteroFloatHost(numero_capas - 1, getCopiaDimensionesCapasRed());
	host_weight_matrices = new GestorPunteroPunteroFloatHost(numero_capas - 1, getCopiaDimensionesMatricesRed());
	host_bias_vectors_fast = host_bias_vectors->getPunteroPunteroHost();
	host_weight_matrices_fast = host_weight_matrices->getPunteroPunteroHost();

	for (int i = 0; i < numero_capas - 1; i++) {
		cudaMemcpy(host_bias_vectors_fast[i], device_bias_vectors->getPunteroPunteroHostDevice()[i], host_bias_vectors->getDimensionesElementos()[i] * sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(host_weight_matrices_fast[i], device_weight_matrices->getPunteroPunteroHostDevice()[i], host_weight_matrices->getDimensionesElementos()[i] * sizeof(float), cudaMemcpyDeviceToHost);
		//imprimirMatrizPorPantalla("bias", host_bias_vectors_fast[i], 1, host_bias_vectors->getDimensionesElementos()[i]);
		//imprimirMatrizPorPantalla("weights", host_weight_matrices_fast[i], dimensiones_capas[i], dimensiones_capas[i+1]);
	}
	//cudaDeviceSynchronize();

	int may = 0;
	for (int i = 1; i < numero_capas; i++) { may = max(may, dimensiones_capas[i]); }
	calc_matrix_fast = new float[may];
	calc_matrix_fast2 = new float[may];
}

const void RedNeuronalSecuencial::propagacionDelanteRapido(const float* input, float* output, int idfun) {
	computeGold(input, host_weight_matrices_fast[0], calc_matrix_fast, 1, dimensiones_capas[0], dimensiones_capas[1]);
	sumarVectores(dimensiones_capas[1], calc_matrix_fast, host_bias_vectors_fast[0]);
	aplicarFuncionHost(idfun, dimensiones_capas[1], calc_matrix_fast);
	for (int i = 1; i < numero_capas - 2; i++) {
		computeGold(calc_matrix_fast, host_weight_matrices_fast[i], calc_matrix_fast2, 1, dimensiones_capas[i], dimensiones_capas[i + 1]);
		memcpy(calc_matrix_fast, calc_matrix_fast2, dimensiones_capas[i + 1]*sizeof(float));
		sumarVectores(dimensiones_capas[i + 1], calc_matrix_fast, host_bias_vectors_fast[i]);
		aplicarFuncionHost(idfun, dimensiones_capas[i + 1], calc_matrix_fast);
	}
	computeGold(calc_matrix_fast, host_weight_matrices_fast[numero_capas - 2], calc_matrix_fast2, 1, dimensiones_capas[numero_capas - 2], dimensiones_capas[numero_capas - 1]);
	sumarVectores(dimensiones_capas[numero_capas - 1], calc_matrix_fast2, host_bias_vectors_fast[numero_capas - 2]);
	memcpy(output, calc_matrix_fast2, dimensiones_capas[numero_capas - 1] * sizeof(float));
	aplicarFuncionHost(idfun, dimensiones_capas[numero_capas - 1], output);
}

void RedNeuronalSecuencial::terminarModoPropagacionDelanteRapido() {
	host_bias_vectors_fast = NULL;
	host_weight_matrices_fast = NULL;
	delete host_bias_vectors;
	delete host_weight_matrices;
}

void RedNeuronalSecuencial::mostrarPesosBiasesRed() {
	float** punteros_biases = device_bias_vectors->getPunteroPunteroHostDevice();
	float** punteros_pesos = device_weight_matrices->getPunteroPunteroHostDevice();
	for (int i = 0; i < numero_capas - 1; i++) {
		printf("\nbias capa %d:\n", i + 1);
		float* h_p = (float*)malloc(dimensiones_capas[i + 1] * sizeof(float));
		cudaMemcpy(h_p, punteros_biases[i], dimensiones_capas[i + 1] * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		imprimirMatrizPorPantalla("", h_p, dimensiones_capas[i + 1], 1);
		free(h_p);
		h_p = NULL;
	}

	for (int i = 0; i < numero_capas - 1; i++) {
		printf("\npesos capa %d:\n", i + 1);
		float* h_p = (float*)malloc(dimensiones_capas[i] * dimensiones_capas[i + 1] * sizeof(float));
		cudaMemcpy(h_p, punteros_pesos[i], dimensiones_capas[i] * dimensiones_capas[i + 1] * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		imprimirMatrizPorPantalla("", h_p, dimensiones_capas[i], dimensiones_capas[i + 1]);
		free(h_p);
		h_p = NULL;
	}
}

void RedNeuronalSecuencial::mostrarZlAl(int batch_size) {
	float** punteros_zl = device_forward_zl->getPunteroPunteroHostDevice();
	float** punteros_al = device_forward_al->getPunteroPunteroHostDevice();
	for (int i = 1; i < numero_capas; i++) {
		printf("\nzl capa %d:\n", i + 1);
		float* h_p = (float*)malloc(dimensiones_capas[i] * batch_size * sizeof(float));
		cudaMemcpy(h_p, punteros_zl[i - 1], dimensiones_capas[i] * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		imprimirMatrizPorPantalla("", h_p, batch_size, dimensiones_capas[i]);
		free(h_p);
		h_p = NULL;
	}

	for (int i = 1; i < numero_capas; i++) {
		printf("\nal capa %d:\n", i + 1);
		float* h_p = (float*)malloc(dimensiones_capas[i] * batch_size * sizeof(float));
		cudaMemcpy(h_p, punteros_al[i - 1], dimensiones_capas[i] * batch_size * sizeof(float), cudaMemcpyDeviceToHost);
		cudaDeviceSynchronize();
		imprimirMatrizPorPantalla("", h_p, batch_size, dimensiones_capas[i]);
		free(h_p);
		h_p = NULL;
	}
}

void RedNeuronalSecuencial::iniciarCublas() {
	cublasCreate(&handle);
}

void RedNeuronalSecuencial::terminarCublas() {
	cublasDestroy(handle);
}