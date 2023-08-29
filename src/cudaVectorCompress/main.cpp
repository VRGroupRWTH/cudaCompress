#include "cudaFuncs.cuh"
#include <vector>
#include <array>
#include <fstream>
#include "ParticleHelpers.h"
#include <algorithm>

std::array<double, 3> CompressionError(const char* filepath_source, const char* filepath_decomp)
{
	// Source dataset
	Vectorfield<float> vf_src{};
	{
		std::fstream file;
		file.open(filepath_source, std::ios::in | std::ios::binary | std::ios::ate);
		if (!file.good())
		{
			return {-1.0, -1.0, -1.0 };
		}
		std::streamsize filesize = file.tellg();
		file.seekg(0, std::ios::beg);
		file.read(reinterpret_cast<char*>(&vf_src.Dimensions), sizeof(vf_src.Dimensions));
		vf_src.ChannelCount = 3;
		vf_src.InterleavedXYZ = false;
		// Debug: Read the dataset into CPU memory
		vf_src.Data.resize(vf_src.NumScalars());
		file.read(reinterpret_cast<char*>(vf_src.Data.data()), filesize - sizeof(vf_src.Dimensions));
	}
	
	// Decompressed dataset
	Vectorfield<float> vf_decomp{};
	{
		std::fstream file;
		file.open(filepath_source, std::ios::in | std::ios::binary | std::ios::ate);
		if (!file.good())
		{
			return { -1.0, -1.0, -1.0 };
		}
		std::streamsize filesize = file.tellg();
		file.seekg(0, std::ios::beg);
		file.read(reinterpret_cast<char*>(&vf_decomp.Dimensions), sizeof(vf_decomp.Dimensions));
		vf_decomp.ChannelCount = 3;
		vf_decomp.InterleavedXYZ = false;
		// Debug: Read the dataset into CPU memory
		vf_decomp.Data.resize(vf_decomp.NumScalars());
		file.read(reinterpret_cast<char*>(vf_decomp.Data.data()), filesize - sizeof(vf_decomp.Dimensions));
	}
}

int main(int argc, char** argv)
{
	//TestCUDAKernel();
	//TestCudaCompressKernel();
	//TestTextureKernel();
	//TestParticleKernel();
	//TestCompressedParticleKernel();

	if (argc < 4) {
		printf("cudaVectorCompress DATASET_PATH SAVE_DECOMPRESSED SAVE_INTERLEAVED [decomposition_levels (2)] [quantization_step_size (0.00136)] [compression_iterations (10)] [huffman_bits (14)]");
		return -1;
	}

	int numDecompLvls = (argc > 4) ? atoi(argv[4]) : 2;
	float quantSize = (argc > 5) ? atof(argv[5]) : 0.00136;
	int compIters = (argc > 6) ? atoi(argv[6]) : 10;
	int huffBits = (argc > 7) ? atoi(argv[7]) : 0;

	CompressFile(argv[1], atoi(argv[2]), atoi(argv[3]), numDecompLvls, quantSize, compIters, huffBits);
	return 0;
}