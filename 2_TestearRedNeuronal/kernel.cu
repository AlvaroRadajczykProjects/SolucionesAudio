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
#define FRAMES_PER_BUFFER 1024 //lo hace 512 veces por segundo? si más grande más datos y más lento aunque a veces puede ser útil

float* devolverDatosEntrenamiento(int* numero_ejemplos, unsigned long long* tam_arr) {

    std::vector<string> archivos = getVectorCharArrayFilesInDirectory(".\\audios_prueba");

    *numero_ejemplos = archivos.size();

    char* narchivo;
    unsigned long long ttotal_entrada = 0;
    unsigned long long ttotal_salida = 0;

    unsigned long long offset_entrada = 0;
    unsigned long long offset_salida = 0;

    //tomar el tamaño total del array a crear
    for (std::vector<string>::iterator it = archivos.begin(); it != archivos.end(); ++it) {

        string cadena_completa = ".\\audios_prueba\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f = new WaveFile(narchivo);

        //calcular
        ttotal_entrada += f->getLenData() + (4096 - (f->getLenData() % 4096));

        free(narchivo);

        delete f;
    }

    *tam_arr = ttotal_entrada / 4;

    float* ret = new float[ttotal_entrada];

    for (std::vector<string>::iterator it = archivos.begin(); it != archivos.end(); ++it) {
        string cadena_completa = ".\\audios_prueba\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f = new WaveFile(narchivo);

        free(narchivo);

        //pasar datos
        memcpy(ret + offset_entrada, (float*)f->getData(), f->getLenData());
        offset_entrada += f->getLenData() / 4;
        while (offset_entrada % 4096 != 0) {
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

    /*

        el formato de los archivos de audio será wave pista mono 32-bit float point (coma flotante), una frecuencia de 44100Hz,
        con la cabecera totalmente limpia (según documentaciones oficiales, de tamaño 44 bytes)

        si quieres ajustar un audio de 48000Hz a 44100Hz, en audacity hay que seleccionar toda la pista, efecto->tono y tempo->
        cambiar la velocidad y cambiar el factor multiplicador por 1,088. Lo exportas y listo. No merece la pena cambiar
        la frecuencia desde audacity, lo hace mal, y luego no se guarda, para eso habría que tocar la cabecera

        en la carpeta audio_source, se deben guardar los audios de voz de la voz de la persona que quiere entrenal el modelo

        en la carpeta audio_target, se deben guardar los audios equivalentes de audio_source pero de la voz que se quiere clonar

        los archivos de audio que tengan el mismo contenido pero distinta voz deben de tener el mismo nombre. Además, deben tener
        el mismo tamaño, sino se entrenará a la red con los datos del más pequeño (es decir, si uno dura 3s y otro 4s, se toma el
        ejemplo  como los 3s enteros del primer audio, y los 3 primeros segundos del segundo audio, se le ha truncado 1s del final

        También, deben haber el mismo número de archivos en ambas carpetas, sino se ignorarán los que no tengan su correspondiente
        archivo pareja con el mimo nombre
    */

    int numero_ejemplos = 0;
    unsigned long long tam_arr = 0;
    float* entrada = 0;

    entrada = devolverDatosEntrenamiento(&numero_ejemplos, &tam_arr);

    //PRUEBAS
    /*
    RedNeuronalSecuencial* r;

    const int nentradas = 2;
    const int nsalidas = 1;
    float tapren = 0.0005;
    int nepochs = 20000;
    float* res;

    const int nejemplos = 4;
    const int batch_size = 4;

    float* de = new float[nentradas * nejemplos] { 0, 0, 0, 1, 1, 0, 1, 1 };
    float* ds = new float[nsalidas * nejemplos] { 0, 1, 1, 0 };

    r = new RedNeuronalSecuencial(4, new int[4] { nentradas, 10, 10, nsalidas }, new int[3] { 3, 3, 3 });

    r->entrenarRedMSE_Adam(tapren, 0.9, 0.999, 0.000000001, 500, nepochs, nejemplos, batch_size, nentradas, nsalidas, de, ds);
    //tapren = 0.0001;
    //r->entrenarRedMSE_SGD(tapren, 100, nepochs, nejemplos, batch_size, nentradas, nsalidas, de, ds);

    r->exportarRedComoArchivo("caca.data");

    r->iniciarCublas();

    res = r->propagacionHaciaDelante(4, nentradas, de);
    imprimirMatrizPorPantalla("", res, 4, nsalidas);
    delete res;

    r->terminarCublas();

    delete r;

    r = new RedNeuronalSecuencial("caca.data");

    //r->mostrarPesosBiasesRed();

    r->iniciarCublas();

    res = r->propagacionHaciaDelante(4, nentradas, de);
    imprimirMatrizPorPantalla("", res, 4, nsalidas);
    delete res;

    r->terminarCublas();

    r->iniciarModoPropagacionDelanteRapido();

    res = new float[1] {0};

    r->propagacionDelanteRapido(new float[2] {0, 0}, res, 2);
    printf("\n%.8f", res[0]);
    r->propagacionDelanteRapido(new float[2] {0, 1}, res, 2);
    printf("\n%.8f", res[0]);
    r->propagacionDelanteRapido(new float[2] {1, 0}, res, 2);
    printf("\n%.8f", res[0]);
    r->propagacionDelanteRapido(new float[2] {1, 1}, res, 2);
    printf("\n%.8f\n", res[0]);

    r->terminarModoPropagacionDelanteRapido();

    delete r;
    */

    RedNeuronalSecuencial* r = new RedNeuronalSecuencial("red.data");
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