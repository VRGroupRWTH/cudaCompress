#include "cudaFuncs.cuh"
#include <vector>
#include <array>
#include <fstream>
#include "ParticleHelpers.h"
#include <algorithm>

/*
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
*/

int main(int argc, char** argv)
{
	//TestCUDAKernel();
	//TestCudaCompressKernel();
	//TestTextureKernel();
	//TestParticleKernel();
	//TestCompressedParticleKernel();

	// if (argc < 3) {
	// 	return -1;
	// }

	int numDecompLvls = (argc > 3) ? atoi(argv[3]) : 2;
	float quantSize = (argc > 4) ? atof(argv[4]) : 0.00136;
	int compIters = (argc > 5) ? atoi(argv[5]) : 10;
	int huffBits = (argc > 6) ? atoi(argv[6]) : 0;

	/*
	uint4 dimensions{ 225, 250, 200, 2 };
	std::vector<float> dataset(dimensions.x * dimensions.y * dimensions.z * dimensions.w * 3UL);

	size_t offset = 0;
	for (int t = 0; t < dimensions.w; t++)
	{
		for (int z = 0; z < dimensions.z; z++)
		{
			for (int y = 0; y < dimensions.y; y++)
			{
				for (int x = 0; x < dimensions.x; x++)
				{
					for (int c = 0; c < 3; c++)
					{
						dataset[offset] = t;
						offset++;
					}
				}
			}
		}
	}

	std::fstream file;
	file.open("small_ds.raw", std::ios::out | std::ios::binary);
	if (!file.good()) return -1;
	file.write(reinterpret_cast<char*>(&dimensions), sizeof(uint4));
	file.write(reinterpret_cast<char*>(dataset.data()), dataset.size() * sizeof(float));
	file.close();
	*/

	uint4 dimensions{ 225, 250, 200, 151 };
	std::vector<float> dataset;
	dataset.resize(size_t(dimensions.x) * size_t(dimensions.y) * size_t(dimensions.z) * size_t(dimensions.w) * 3ULL);

	std::vector<float> slice;
	slice.resize(size_t(dimensions.x) * size_t(dimensions.y) * size_t(dimensions.z) * 20ULL * 4ULL);

	size_t channel_offset = (size_t)dimensions.w * (size_t)dimensions.z * (size_t)dimensions.y * (size_t)dimensions.x;
	for (auto i = 1; i < argc; i++)
	{
		std::fstream file;
		file.open(argv[i], std::ios::in | std::ios::binary | std::ios::ate);
		if (!file.good())
		{
			printf("Error reading file");
			return -1.0;
		}
		auto filesize = file.tellg();
		file.seekg(0, std::ios::beg);
		file.read(reinterpret_cast<char*>(slice.data()), filesize);
		file.close();

		for (size_t s = 0; s < 20; s++)
		{
			size_t time_offset = (i - 1UL) * 20UL;
			size_t curr_time = s + time_offset;
			printf("t: %u\n", curr_time);

			if (curr_time > 150)
			{
				printf("Finished");
				break;
			}

			for (size_t z = 0; z < dimensions.z; z++)
			{
				for (size_t y = 0; y < dimensions.y; y++)
				{
					for (size_t x = 0; x < dimensions.x; x++)
					{
						size_t idx_dataset = ((s + time_offset) * (size_t)dimensions.z * (size_t)dimensions.y * (size_t)dimensions.x
							+ z * (size_t)dimensions.y * (size_t)dimensions.x
							+ y * (size_t)dimensions.x
							+ x);

						size_t idx_slice = 
							(s * (size_t)dimensions.z * (size_t)dimensions.y * (size_t)dimensions.x
							+ z * (size_t)dimensions.y * (size_t)dimensions.x
							+ y * (size_t)dimensions.x
							+ x) * 4UL;

						size_t idx_slice2 = 
							(z * (size_t)dimensions.y * (size_t)dimensions.x * 20UL
							+ y * (size_t)dimensions.x * 20UL
							+ x * 20UL
							+ s) * 4UL;

						dataset[idx_dataset + 0UL * channel_offset] = slice[idx_slice2 + 0UL];
						dataset[idx_dataset + 1UL * channel_offset] = slice[idx_slice2 + 1UL];
						dataset[idx_dataset + 2UL * channel_offset] = slice[idx_slice2 + 2UL];
					}
				}
			}
		}

		printf("%f\n", dataset[dataset.size() - 1UL]);
		printf("%f\n", dataset[dataset.size() - 2UL]);
		printf("%f\n", dataset[dataset.size() - 3UL]);
	}

	printf("\n");
	printf("%f\n", dataset[0]);
	printf("%f\n", dataset[channel_offset]);
	printf("%f\n", dataset[2UL * channel_offset]);

	printf("Writing file");
	std::fstream file;
	file.open("abc_vulkan.raw", std::ios::out | std::ios::binary);
	if (!file.good())
	{
		return -1.0;
	}
	file.write(reinterpret_cast<char*>(&dimensions), sizeof(dimensions));
	file.write(reinterpret_cast<char*>(dataset.data()), dataset.size() * sizeof(float));
	file.close();


	//TraceABC();
	//TraceFile(argv[1]);
	//CompressFile(argv[1], true, argv[2], numDecompLvls, quantSize, compIters, huffBits);
	//ReadFile(argv[1]);
	return 0;
}