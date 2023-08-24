#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <iostream>
#include <ctime>
#include <chrono>
#include <cstring>

#include <portaudio.h>

#include "RedNeuronalSecuencial.cuh"

#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 256 //lo hace 512 veces por segundo? si más grande más datos y más lento aunque a veces puede ser útil

using namespace std;

RedNeuronalSecuencial* r;

__global__ void processAudioKernel(const float* input, float* output)
{
	output[threadIdx.x] = input[threadIdx.x];
}

static int patestCallback(
	const void* inputBuffer,
	void* outputBuffer,
	unsigned long framesPerBuffer,
	const PaStreamCallbackTimeInfo* timeInfo,
	PaStreamCallbackFlags statusFlags,
	void* userData
) {

	float* ve = (float*)inputBuffer;
	float* v3 = (float*)outputBuffer;

	r->propagacionDelanteRapido(ve, v3, 2, 3);

	//float* pred = r->propagacionHaciaDelante(1, FRAMES_PER_BUFFER, ve);
	//cudaDeviceSynchronize();
	//memcpy(v3, pred, FRAMES_PER_BUFFER * sizeof(float));
	//free(pred);
		
	//for (int i = 0; i < 1024; i++) { v3[i] = (i + 1) / (float)1024; }

	//memcpy(v3, ve, FRAMES_PER_BUFFER * sizeof(float));

	return paContinue;
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

void showDevices(int numDevices) {
	const PaDeviceInfo* deviceInfo;
	for (int i = 0; i < numDevices; i++) {
		deviceInfo = Pa_GetDeviceInfo(i);
		printf("Device %d:\n", i);
		printf("    name %s:\n", deviceInfo->name);
		printf("    maxInputChannels %d:\n", deviceInfo->maxInputChannels);
		printf("    maxOutputChannels %d:\n", deviceInfo->maxOutputChannels);
		printf("    defaultSampleRate %f:\n", deviceInfo->defaultSampleRate);
	}
}

int main(int argc, char** argv) {

	r = new RedNeuronalSecuencial("..\\red.data");
	cudaDeviceSynchronize();
	r->iniciarModoPropagacionDelanteRapido();

	PaError err;
	err = Pa_Initialize();
	checkErr(err);

	int numDevices = Pa_GetDeviceCount();
	manageNumberDevices(numDevices);

	PaStream* stream;
	err = Pa_OpenDefaultStream(&stream, 1, 1, paFloat32, SAMPLE_RATE, FRAMES_PER_BUFFER, patestCallback, nullptr);
	checkErr(err);

	err = Pa_StartStream(stream);
	checkErr(err);

	Pa_Sleep(100 * 1000);

	err = Pa_StopStream(stream);
	checkErr(err);

	err = Pa_CloseStream(stream);
	checkErr(err);

	err = Pa_Terminate();
	checkErr(err);
	
	r->terminarModoPropagacionDelanteRapido();
	delete r;

	return EXIT_SUCCESS;
}