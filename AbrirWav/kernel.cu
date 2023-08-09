#include <iostream>
#include <string>
#include <fstream>
#include <cstdint>

#include <portaudio.h>

#define SAMPLE_RATE 44100 //48000
#define FRAMES_PER_BUFFER 1024 //lo hace 512 veces por segundo? si más grande más datos y más lento aunque a veces puede ser útil

// find the file size
int getFileSize(FILE* inFile)
{
    int fileSize = 0;
    fseek(inFile, 0, SEEK_END);

    fileSize = ftell(inFile);

    fseek(inFile, 0, SEEK_SET);
    return fileSize;
}

FILE* loadWavFile(const char* filePath) {
    FILE* wavFile = fopen(filePath, "r");
    if (wavFile == nullptr)
    {
        fprintf(stderr, "Unable to open wave file: %s\n", filePath);
        exit(EXIT_FAILURE);
    }
    return wavFile;
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

int main(int argc, char* argv[])
{
    FILE* wavFile = loadWavFile("audio2.wav");

    int fsize = getFileSize(wavFile);
    int wav_header_size_bytes = 44;

    //lo único con lo que hay que tener cuidado es con que la cabecera de un archivo wav mide 44 bytes, el resto hasta el final son datos,

    /*printf("Tam de archivo: %d\n", fsize);
    printf("Tam de los datos: %d\n", fsize - wav_header_size_bytes);
    printf("Numero de numeros en coma flotante: %d\n", (fsize - wav_header_size_bytes)/4);
    printf("Sobran bytes?: %d\n", (fsize - wav_header_size_bytes)%4);
    printf("Primera posicion: %d\n", wav_header_size_bytes/4);
    printf("Ultima posicion: %d\n", ((fsize/4)-1) );*/

    char* buffer = (char*)malloc(fsize + 1);
    fread(buffer, 1, fsize, wavFile);
    fclose(wavFile);

    float* fbuff = (float*)buffer;

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

    for (int i = wav_header_size_bytes / 4; i < (fsize / 4); i += FRAMES_PER_BUFFER) {
        err = Pa_WriteStream(stream, fbuff + i, FRAMES_PER_BUFFER);
        checkErr(err);
    }

    err = Pa_StopStream(stream);
    checkErr(err);

    err = Pa_CloseStream(stream);
    checkErr(err);

    err = Pa_Terminate();
    checkErr(err);

    return 0;
}

