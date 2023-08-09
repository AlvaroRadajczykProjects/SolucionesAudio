#include "funciones_archivos.h"

using namespace std;
namespace fs = std::filesystem;

std::vector<string> getVectorCharArrayFilesInDirectory(const char* directorio) {
    fs::path p(directorio);
    std::vector<string> s;

    for (auto i = fs::directory_iterator(p); i != fs::directory_iterator(); i++)
    {
        std::string cadenastr = i->path().filename().string().c_str();
        s.push_back(cadenastr);
    }

    return s;
}

std::vector<string> intersection(std::vector<string> v1, std::vector<string> v2) {
    std::vector<string> v3;

    std::sort(v1.begin(), v1.end());
    std::sort(v2.begin(), v2.end());

    std::set_intersection(v1.begin(), v1.end(),
        v2.begin(), v2.end(),
        back_inserter(v3));
    return v3;
}

void showVectorCharArray(std::vector<string> s) {
    for (std::vector<string>::iterator it = s.begin(); it != s.end(); ++it) {
        std::string cadena = *it;
        std::cout << ' ' << cadena;
    }
    std::cout << '\n';
}

char* stdStringAPunteroChar(string s) {
    char* p = (char*) malloc( (s.length()+1)*sizeof(char) );
    p[s.length()] = NULL;
    strcpy(p, s.c_str());
    return p;
}

unsigned long long getFileSize(FILE* inFile)
{
    unsigned long long fileSize = 0;
    fseek(inFile, 0, SEEK_END);

    fileSize = ftell(inFile);

    fseek(inFile, 0, SEEK_SET);
    return fileSize;
}

FILE* cargarArchivo(const char* filePath) {
    FILE* file = fopen(filePath, "r");
    if (file == nullptr)
    {
        fprintf(stderr, "Unable to open file: %s\n", filePath);
        exit(EXIT_FAILURE);
    }
    return file;
}

void crearArchivoEscribirYCerrar(const char* nombre, int nbytes, char* dbytes) {
    FILE* f = fopen(nombre, "wb");
    fwrite(dbytes, 1, nbytes, f);
    fclose(f);
}

//en tam, poner la dirección de memoria de un entero
char* leerArchivoYCerrar(const char* nombre, unsigned int* tam) {
    FILE* f = fopen(nombre, "rb");
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);

    char* string = (char*)malloc(fsize);
    fread(string, 1, fsize, f);
    fclose(f);

    return string;
}

float* devolverArrayFloatDatosTodosArchivosAudio(string directorio, unsigned long long* tam) {
    
    std::vector<string> archivos = intersection(
        getVectorCharArrayFilesInDirectory(".\\audio_source"),
        getVectorCharArrayFilesInDirectory(".\\audio_target")
    );

    char* narchivo;
    unsigned long long ttotal = 0;

    //tomar el tamaño total del array a crear
    for (std::vector<string>::iterator it = archivos.begin(); it != archivos.end(); ++it) {
        string cadena_completa = ".\\audio_source\\" + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        FILE* f = fopen(narchivo, "rb");
        unsigned long long tarch = getFileSize(f);
        fclose(f);

        ttotal += ( (unsigned long long) ceil( (tarch - 44) / (float) (4*1024) ) ) * 1024; //( ( (unsigned long long) ceil( (tarch-44) / (float) 1024 ) ) * 1024 ) / 4;

        free(narchivo);
        narchivo = NULL;
    }

    //crear el puntero
    *tam = ttotal;
    float* ret = new float[ttotal];

    unsigned long long offset = 0;
    //cargar los datos al puntero
    for (std::vector<string>::iterator it = archivos.begin(); it != archivos.end(); ++it) {
        string cadena_completa = directorio + *it;
        narchivo = stdStringAPunteroChar(cadena_completa);

        FILE* f = fopen(narchivo, "rb");
        if (!f) { printf("\nError al abrir el archivo %s\n", narchivo); }
        unsigned long long tarch = (getFileSize(f) - 44) / 4;

        fseek(f, 60, SEEK_SET);
        fread(ret+offset, 4, tarch, f);
        offset += tarch;
        //printf("\noffset: %u %u\n", offset, offset%1024);
        while (offset % 1024 != 0) { ret[offset] = 0.0; offset++; }
        //printf("\noffset: %u %u\n", offset, offset % 1024);

        //printf("%s terminado\n", narchivo);
        free(narchivo);
        narchivo = NULL;
    }

    return ret;

}