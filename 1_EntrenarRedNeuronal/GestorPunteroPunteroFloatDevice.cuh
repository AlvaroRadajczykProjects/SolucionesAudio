#include "basicos.cuh"

class GestorPunteroPunteroFloatDevice {

	private:
		int numero_elementos;
		int* dimensiones_elementos;
		float** hd_p;
		float** d_p;

	public:
		GestorPunteroPunteroFloatDevice(int ne, int* de);
		~GestorPunteroPunteroFloatDevice();
		int getNumeroElementos();
		int* getDimensionesElementos();
		float** getPunteroPunteroDevice();
		float** getPunteroPunteroHostDevice();
		void copiarHostADevice(float** h_p);
		void copiarDeviceAHost(float** h_p);
		void ponerElementosTodosElementosACero();

};

