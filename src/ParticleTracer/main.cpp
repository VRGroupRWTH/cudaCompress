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

	if (argc < 3) {
		return -1;
	}

	int numDecompLvls = (argc > 3) ? atoi(argv[3]) : 2;
	float quantSize = (argc > 4) ? atof(argv[4]) : 0.00136;
	int compIters = (argc > 5) ? atoi(argv[5]) : 10;
	int huffBits = (argc > 6) ? atoi(argv[6]) : 0;

	/*
	uint4 Dimensions{ 0, 0, 0, 0 };
	std::fstream file;
	file.open(argv[1], std::ios::in | std::ios::binary | std::ios::ate);
	if (!file.good())
	{
		return -1;
	}
	std::streamsize filesize = file.tellg();
	file.seekg(0, std::ios::beg);
	file.read(reinterpret_cast<char*>(&Dimensions), sizeof(Dimensions));

	Vectorfield<float> vf_src{};
	vf_src.Dimensions = Dimensions;
	vf_src.ChannelCount = 3;
	vf_src.InterleavedXYZ = false;
	// Debug: Read the dataset into CPU memory
	vf_src.Data.resize(vf_src.NumScalars());
	file.read(reinterpret_cast<char*>(vf_src.Data.data()), filesize - sizeof(Dimensions));

	uint4 Dimensions_new {230, 250, 200, 151};
	std::vector<float> vf_new;
	vf_new.resize((size_t)Dimensions_new.x * (size_t)Dimensions_new.y * (size_t)Dimensions_new.z * (size_t)Dimensions_new.w * 3UL);

	size_t offset_old = 0;
	size_t offset_new = 0;
	for (auto t = 0; t < Dimensions_new.w; t++)
	{
		for (auto z = 0; z < Dimensions_new.z; z++)
		{
			for (auto y = 0; y < Dimensions_new.y; y++)
			{
				for (auto x = 0; x < Dimensions_new.x; x++)
				{
					offset_old =
						( (size_t)(std::min((uint)t, Dimensions.w - 1)) * Dimensions.x * Dimensions.y * Dimensions.z
						+ (size_t)(std::min((uint)z, Dimensions.z - 1)) * Dimensions.y * Dimensions.x
						+ (size_t)(std::min((uint)y, Dimensions.y - 1)) * Dimensions.x
						+ (size_t)(std::min((uint)x, Dimensions.x - 1))
						) * 3UL;

					vf_new[offset_new + 0U] = vf_src.Data[offset_old + 0U];
					vf_new[offset_new + 1U] = vf_src.Data[offset_old + 1U];
					vf_new[offset_new + 2U] = vf_src.Data[offset_old + 2U];

					offset_new += 3;
				}
			}
		}
	}

	std::fstream file_out;
	file_out.open("abc_padded.raw", std::ios::out | std::ios::binary);
	file_out.write(reinterpret_cast<char*>(&Dimensions_new), sizeof(Dimensions_new));
	file_out.write(reinterpret_cast<char*>(vf_new.data()), vf_new.size() * sizeof(float));
	file_out.close();
	*/

	//TraceABC();
	//TraceFile(argv[1]);
	CompressFile(argv[1], true, argv[2], numDecompLvls, quantSize, compIters, huffBits);
	return 0;
}