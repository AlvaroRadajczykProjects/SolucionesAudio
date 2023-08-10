#include "file_manager.h"

using namespace std;

char* returnFileData( const char* name, streampos* size ) {
    char* memblock;
    ifstream file(name, ios::in | ios::binary | ios::ate);
    if (file.is_open()) {
        *size = file.tellg();
        memblock = new char[*size];
        file.seekg(0, ios::beg);
        file.read(memblock, *size);
        file.close();
        return memblock;
    }
    return NULL;
}