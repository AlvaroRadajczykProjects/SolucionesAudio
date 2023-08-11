#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <windows.h>
#include <iostream>
#include <string.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "GestorPunteroPunteroFloatHost.cuh"
#include "GestorPunteroPunteroFloatDevice.cuh"

#include "funciones_archivos.h"

void imprimirVectorIntPorPantalla(char* texto_mostrar, float vector[], int inicio, int fin);
void imprimirMatrizPorPantalla(char* texto_mostrar, float matriz[], int n_filas, int n_columnas);

const void matrizTraspuestaDevice(float* odata, float* idata, int m, int n);
const void normalizar(float* data, unsigned long long length);

class RedNeuronalSecuencial {

	private:
		int numero_capas;
		int* dimensiones_capas = NULL;
		int* funciones_capas = NULL;

		//red en device, se guardan los vectores de bias y las matrices de pesos, por tanto no se guarda la capa de entrada
		//ojo, si se quiere serializar esta clase, hay que eliminar bien todito
		GestorPunteroPunteroFloatDevice* device_bias_vectors = NULL;
		GestorPunteroPunteroFloatDevice* device_weight_matrices = NULL;

		//matriz que guarda el array de valores de entrada en el device
		float* device_batch_input = NULL;
		//matriz que guarda el array de valores de salida en el device
		float* device_batch_output = NULL;

		//matriz en el device donde se guarda temporalmente una matriz para calcular matrices traspuestas
		float* temp_matr_traspose = NULL;

		//matrices donde se guardan los valores de la propagación hacia adelante necesario para el entrenamiento
		//en al, al hacer backpropagation, se guardan los valores de los errores de los biases para todos los ejemplos
		GestorPunteroPunteroFloatDevice* device_forward_zl = NULL;
		GestorPunteroPunteroFloatDevice* device_forward_al = NULL;

		//matrices donde se guardan los valores de error de los pesos y de los biases de los vectores gradientes
		GestorPunteroPunteroFloatDevice* device_err_bias_vgrad = NULL;
		GestorPunteroPunteroFloatDevice* device_err_weight_vgrad = NULL;

		//aquí se almacenan los valores de momento y velocidad
		GestorPunteroPunteroFloatDevice* device_err_bias_m = NULL;
		GestorPunteroPunteroFloatDevice* device_err_weight_m = NULL;
		GestorPunteroPunteroFloatDevice* device_err_bias_v = NULL;
		GestorPunteroPunteroFloatDevice* device_err_weight_v = NULL;

		//para el modo rápido
		GestorPunteroPunteroFloatHost* host_bias_vectors;
		GestorPunteroPunteroFloatHost* host_weight_matrices;

		float** host_bias_vectors_fast;
		float** host_weight_matrices_fast;
		float* calc_matrix_fast;
		float* calc_matrix_fast2;

		int* getCopiaDimensionesCapasRed();
		int* getCopiaDimensionesMatricesRed();
		int getMaxTamMatrTempTrans(int batch_size);
		int* getDimensionesZlAl(int batch_size);

		void cargarEnDevice(bool iniciarValoresBiasesWeights);
		void copiarPesosHostDevice(float** host_pesos, float** host_biases);

		float calcularFuncionCosteMSE(int batch_size, int nvalsalida, float* vsalida);

		void calcularVectorGradiente(int batch_size, int nvalsalida, float* vsalida);

		void aplicarVectorGradienteSGD(float tapren, int batch_size);
		void aplicarVectorGradienteAdam(float tapren, float b1, float b2, float epsilon, int batch_size);

		void propagacionHaciaDelanteEntrenamiento(int nejemplos, int nvalsentrada, float* matrizejemplos);

	public:
		RedNeuronalSecuencial(int nc, int* dc, int* fc);
		RedNeuronalSecuencial(const char* nombre_archivo);
		void cargarPunterosHostBiasesWeights(GestorPunteroPunteroFloatHost* biases, GestorPunteroPunteroFloatHost* weights);
		~RedNeuronalSecuencial();
		int getNumeroCapas();
		int* getDimensionesCapas();
		int* getFuncionesCapas();
		void exportarRedComoArchivo(const char* nombre_archivo);
		float* propagacionHaciaDelante(int nejemplos, int nvalsentrada, float* matrizejemplos);

		void entrenarRedMSE_SGD(float tapren, int mostrar_fcoste_cada_n_epocas, int nepocas, int nejemplos, int batch_size, int nvalsentrada, int nvalssalida, float* ventrada, float* vsalida);
		void entrenarRedMSE_Adam(float tapren, float b1, float b2, float epsilon, int mostrar_fcoste_cada_n_epocas, int nepocas, int nejemplos, int batch_size, int nvalsentrada, int nvalsalida, float* ventrada, float* vsalida);

		void iniciarModoPropagacionDelanteRapido();
		const void propagacionDelanteRapido(const float* input, float* output, int idfunco, int idfuncs);
		void terminarModoPropagacionDelanteRapido();

		void iniciarCublas();
		void terminarCublas();

		void mostrarPesosBiasesRed();
		void mostrarZlAl(int batch_size);

};