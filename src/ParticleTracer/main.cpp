#include "cudaTests.cuh"

int main(int argc, char** argv)
{
	//TestCUDAKernel();
	//TestCudaCompressKernel();
	//TestTextureKernel();
	//TestParticleKernel();
	TestCompressedParticleKernel();

	return 0;
}