#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cstring>

#include <portaudio.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define SAMPLE_RATE 44100
#define FRAMES_PER_BUFFER 1024 //lo hace 512 veces por segundo? si más grande más datos y más lento aunque a veces puede ser útil

float* d_input;
float* d_output;

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

	float* input = (float*)inputBuffer;
	float* output = (float*)outputBuffer;

	cudaMemcpy(d_input, input, FRAMES_PER_BUFFER * sizeof(float), cudaMemcpyHostToDevice);
	processAudioKernel << <1, FRAMES_PER_BUFFER >> > (d_input, d_output);
	cudaMemcpy(output, d_output, FRAMES_PER_BUFFER * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

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

	cudaMalloc(&d_input, FRAMES_PER_BUFFER * sizeof(float));
	cudaMalloc(&d_output, FRAMES_PER_BUFFER * sizeof(float));
	cudaDeviceSynchronize();

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

	return EXIT_SUCCESS;
}