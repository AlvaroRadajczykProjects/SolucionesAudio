#include <windows.h>

#include <stdio.h>
#include <stdlib.h>

#include <windows.h>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

std::vector<string> getVectorCharArrayFilesInDirectory(const char* directorio);
std::vector<string> intersection(std::vector<string> v1, std::vector<string> v2);
void showVectorCharArray(std::vector<string> s);
char* stdStringAPunteroChar(string s);

unsigned long long getFileSize(FILE* inFile);
FILE* cargarArchivo(const char* filePath);
void crearArchivoEscribirYCerrar(const char* nombre, int nbytes, char* dbytes);
char* leerArchivoYCerrar(const char* nombre, unsigned int* tam);

float* devolverArrayFloatDatosTodosArchivosAudio(string directorio, unsigned long long* tam);