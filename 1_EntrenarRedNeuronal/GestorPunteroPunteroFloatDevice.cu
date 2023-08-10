#include "GestorPunteroPunteroFloatDevice.cuh"

GestorPunteroPunteroFloatDevice::GestorPunteroPunteroFloatDevice(int ne, int* de) {
	numero_elementos = ne;
	dimensiones_elementos = de;

	hd_p = (float**)malloc(numero_elementos * sizeof(float*));
	manageCUDAError(cudaMalloc(&d_p, numero_elementos * sizeof(float*)));

	for (int i = 0; i < numero_elementos; i++) {
		manageCUDAError(cudaMalloc(&hd_p[i], dimensiones_elementos[i] * sizeof(float)));
	}

	manageCUDAError(cudaMemcpy(d_p, hd_p, numero_elementos * sizeof(float*), cudaMemcpyHostToDevice));

	manageCUDAError(cudaDeviceSynchronize());
}

GestorPunteroPunteroFloatDevice::~GestorPunteroPunteroFloatDevice() {
	for (int i = 0; i < numero_elementos; i++) {
		manageCUDAError(cudaFree(hd_p[i]));
		hd_p[i] = NULL;
	}
	manageCUDAError(cudaFree(d_p));
	d_p = NULL;
	free(hd_p);
	hd_p = NULL;
	free(dimensiones_elementos);
	dimensiones_elementos = NULL;
}

int GestorPunteroPunteroFloatDevice::getNumeroElementos() {
	return numero_elementos;
}

int* GestorPunteroPunteroFloatDevice::getDimensionesElementos() {
	return dimensiones_elementos;
}

float** GestorPunteroPunteroFloatDevice::getPunteroPunteroDevice() {
	return d_p;
}

float** GestorPunteroPunteroFloatDevice::getPunteroPunteroHostDevice() {
	return hd_p;
}

void GestorPunteroPunteroFloatDevice::copiarHostADevice(float** h_p) {

	for (int i = 0; i < numero_elementos; i++) {
		manageCUDAError(cudaMemcpy(hd_p[i], h_p[i], dimensiones_elementos[i] * sizeof(float), cudaMemcpyHostToDevice));
	}
	manageCUDAError(cudaDeviceSynchronize());
}

void GestorPunteroPunteroFloatDevice::copiarDeviceAHost(float** h_p) {
	for (int i = 0; i < numero_elementos; i++) {
		manageCUDAError(cudaMemcpy(h_p[i], hd_p[i], dimensiones_elementos[i] * sizeof(float), cudaMemcpyDeviceToHost));
	}
	manageCUDAError(cudaDeviceSynchronize());
}

void GestorPunteroPunteroFloatDevice::ponerElementosTodosElementosACero() {
	for (int i = 0; i < numero_elementos; i++) {
		manageCUDAError(cudaMemset((void*) hd_p[i], 0, dimensiones_elementos[i] * sizeof(float)));
		manageCUDAError(cudaDeviceSynchronize());
		//ponerTodosElementosVectorCero << < ( (int) ceil( dimensiones_elementos[i] / (float) 1024 ) ), 1024 >> > (hd_p[i], dimensiones_elementos[i]);
		//manageCUDAError(cudaDeviceSynchronize());
	}
}

