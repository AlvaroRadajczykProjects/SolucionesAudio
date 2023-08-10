#include "RedNeuronalSecuencial.cuh"
#include "funciones_archivos.h"
#include "WaveFile.h"

#include <stdio.h>
#include <stdlib.h>

#include <time.h>
#include <curand.h>
#include <curand_kernel.h>

#include <portaudio.h>

#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 512 //lo hace 512 veces por segundo? si más grande más datos y más lento aunque a veces puede ser útil

float* devolverDatosEntrenamiento(int* numero_ejemplos, unsigned long long* tam_arr) {

    std::vector<string> archivos = getVectorCharArrayFilesInDirectory("..\\audio_source\\");

    *numero_ejemplos = archivos.size();

    char* narchivo;
    unsigned long long ttotal_entrada = 0;
    unsigned long long ttotal_salida = 0;

    unsigned long long offset_entrada = 0;
    unsigned long long offset_salida = 0;

    int tbytes_segmento = FRAMES_PER_BUFFER * 4;

    //tomar el tamaño total del array a crear
    for (std::vector<string>::iterator it = archivos.begin(); it != archivos.end(); ++it) {

        string cadena_completa = "..\\audio_source\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f = new WaveFile(narchivo);

        //calcular
        ttotal_entrada += f->getLenData() + (tbytes_segmento - (f->getLenData() % tbytes_segmento));

        free(narchivo);

        delete f;
    }

    *tam_arr = ttotal_entrada / 4;

    float* ret = new float[ttotal_entrada];

    for (std::vector<string>::iterator it = archivos.begin(); it != archivos.end(); ++it) {
        string cadena_completa = "..\\audio_source\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f = new WaveFile(narchivo);

        free(narchivo);

        //pasar datos
        memcpy(ret + offset_entrada, (float*)f->getData(), f->getLenData());
        offset_entrada += f->getLenData() / 4;
        while (offset_entrada % FRAMES_PER_BUFFER != 0) {
            ret[offset_entrada] = 0.0;
            offset_entrada++;
        }

        delete f;
    }

    return ret;

}

static void checkErr(PaError err) {
    if (err != paNoError) {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        exit(EXIT_FAILURE);
    }
}

void manageNumberDevices(int numDevices) {
    printf("Number of devices: %d\n", numDevices);
    if (numDevices < 0) {
        printf("Error getting device count.\n");
        exit(EXIT_FAILURE);
    }
    else if (numDevices < 1) {
        printf("No record devices available.\n");
        exit(EXIT_FAILURE);
    }
}

int main() {

    srand(time(NULL));

    int numero_ejemplos = 0;
    unsigned long long tam_arr = 0;
    float* entrada = 0;

    entrada = devolverDatosEntrenamiento(&numero_ejemplos, &tam_arr);

    RedNeuronalSecuencial* r = new RedNeuronalSecuencial("..\\red.data");
    cudaDeviceSynchronize();
    r->iniciarModoPropagacionDelanteRapido();

    PaError err;
    err = Pa_Initialize();
    checkErr(err);

    int numDevices = Pa_GetDeviceCount();
    manageNumberDevices(numDevices);

    PaStream* stream;
    err = Pa_OpenDefaultStream(&stream, 0, 1, paFloat32, SAMPLE_RATE, FRAMES_PER_BUFFER, nullptr, nullptr);
    checkErr(err);

    err = Pa_StartStream(stream);
    checkErr(err);

    float* buffer = new float[FRAMES_PER_BUFFER];

    for (int i = 0; i < tam_arr; i += FRAMES_PER_BUFFER) {
        r->propagacionDelanteRapido(entrada + i, buffer, 1);
        err = Pa_WriteStream(stream, buffer, FRAMES_PER_BUFFER);
        checkErr(err);
    }

    err = Pa_StopStream(stream);
    checkErr(err);

    err = Pa_CloseStream(stream);
    checkErr(err);

    err = Pa_Terminate();
    checkErr(err);

    r->terminarModoPropagacionDelanteRapido();
    delete r;

    return 0;
}