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
#define FRAMES_PER_BUFFER 256 //lo hace 512 veces por segundo? si más grande más datos y más lento aunque a veces puede ser útil

struct pointers {
    float* ptr1;
    float* ptr2;
};

struct pointers devolverDatosEntrenamiento( int* numero_ejemplos, unsigned long long* tam_arr ) {

    struct pointers p;

    std::vector<string> archivos = intersection(
        getVectorCharArrayFilesInDirectory("..\\audio_source"),
        getVectorCharArrayFilesInDirectory("..\\audio_target")
    );

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

        free(narchivo);
        cadena_completa = "..\\audio_target\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f2 = new WaveFile(narchivo);

        //calcular
        ttotal_entrada += f->getLenData() + (tbytes_segmento - (f->getLenData()%tbytes_segmento) );
        ttotal_salida += f2->getLenData() + (tbytes_segmento - (f2->getLenData()%tbytes_segmento) );

        if (ttotal_entrada != ttotal_salida) {
            printf("\nError al cargar datos: tienen un tiempo de duracion distintos los archivos con el nombre %s\n", narchivo);
            exit(EXIT_FAILURE);
        }

        free(narchivo);

        delete f;
        delete f2;
    }

    *tam_arr = ttotal_entrada/4;

    p.ptr1 = new float[ttotal_entrada];
    p.ptr2 = new float[ttotal_salida];

    for (std::vector<string>::iterator it = archivos.begin(); it != archivos.end(); ++it) {
        string cadena_completa = "..\\audio_source\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f = new WaveFile(narchivo);

        free(narchivo);
        cadena_completa = "..\\audio_target\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        WaveFile* f2 = new WaveFile(narchivo);

        free(narchivo);

        //pasar datos
        memcpy(p.ptr1 + offset_entrada, (float*)f->getData(), f->getLenData());
        offset_entrada += f->getLenData()/4;
        while (offset_entrada % FRAMES_PER_BUFFER != 0) {
            p.ptr1[offset_entrada] = 0.0;
            offset_entrada++;
        }

        memcpy(p.ptr2 + offset_salida, (float*)f2->getData(), f2->getLenData());
        offset_salida += f2->getLenData()/4;
        while (offset_salida % FRAMES_PER_BUFFER != 0) {
            p.ptr2[offset_salida] = 0.0;
            offset_salida++;
        }

        delete f;
        delete f2;
    }

    return p;

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
    float* salida = 0;

    struct pointers p;
    p = devolverDatosEntrenamiento(&numero_ejemplos, &tam_arr);

    entrada = p.ptr1;
    salida = p.ptr2;

    printf("\nNumero de ejemplos: %d\n", numero_ejemplos);

    //printf("\nNormalizando conjunto de entrenamiento...\n");

    //normalizar(entrada, tam_arr/4);

    //printf("\nconjunto de entrenamiento normalizado\n");

    const int nentradas = FRAMES_PER_BUFFER;
    const int nsalidas = FRAMES_PER_BUFFER;
    float tapren = 0.0009;
    int nepochs = 10000;

    const int nejemplos = tam_arr / FRAMES_PER_BUFFER;//tam_arr/1024;//tam_arr%1024;
    const int batch_size = 65536;//65536;

    RedNeuronalSecuencial* r;

    //entrenar desde 0
    r = new RedNeuronalSecuencial(4, new int[4] { nentradas, FRAMES_PER_BUFFER, FRAMES_PER_BUFFER, nsalidas }, new int[3] { 5, 5, 4 });

    //entrenar desde archivo
    //r = new RedNeuronalSecuencial("..\\red.data");

    //un error de 0,00005 es bastante bueno
    for (int i = 0; i < 5; i++) {
        printf("\n\n=============================== FASE %d ===============================\n\n", i + 1);
        //r->entrenarRedMSE_SGD(tapren, 100, nepochs, nejemplos, batch_size, nentradas, nsalidas, entrada, salida);
        r->entrenarRedMSE_Adam(tapren, 0.9, 0.999, 0.000000001, 500, nepochs, nejemplos, batch_size, nentradas, nsalidas, entrada, salida);
        if(tapren > 0.0001){ tapren -= 0.0001; }
        //tapren = tapren / 2;
        r->exportarRedComoArchivo("..\\red.data");
    }

    delete r;

    return 0;
}