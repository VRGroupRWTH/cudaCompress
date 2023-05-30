#include "Tests.h"

#include "CompressVolume.h"
#include "Datasets.h"
#include "cudaUtil.h"
#include "cudaTests.cuh"

#include <glm/vec3.hpp>
#include <glm/gtx/compatibility.hpp>

/*
int TestCudaCompress()
{
	int DimX = 156;
	int DimY = 256;
	int DimZ = 256;
	int DimT = 1;

	size_t Elems2D = DimX * DimY;
	size_t Elems3D = DimX * DimY * DimZ;
	size_t Bytes2D = Elems2D * sizeof(float);
	size_t Bytes3D = Elems3D * sizeof(float);

	// Dataset description
	const uint elemCountTotal = DimX * DimY * DimZ;		// Vectors in volume
	const size_t channelCount = 4;						// Components in vector

	// Create Dataset
	//auto dataset = GenerateTestDataset<float>(DimX, DimY, DimZ, DimT, channelCount, false);
	auto dataset = GenerateABCDataset<float>(DimX, DimY, DimZ, DimT, channelCount, false);
	auto compressed_dataset = dataset;

	// Compression Settings
	const bool doRLEOnlyOnLvl0 = true;					// [?] Dont know why
	const uint8_t numLevels = 2;
	const float quantStep = 0.00136f;
	uint8_t iterations = 10;

	//printf("Press Enter to continue... (1) \n");
	//getchar();

	// Allocate GPU arrays and upload data
	std::vector<float*> dp_Buffer_Images(channelCount);			// Apparently 1 GPU buffer per vector component
	for (size_t c = 0; c < channelCount; c++)
	{
		const auto channelOffset = c * DimX * DimY * DimZ * DimT;
		// Allocate Memory on GPU
		cudaSafeCall(cudaMalloc(&dp_Buffer_Images[c], elemCountTotal * sizeof(float)));
		// Copy memory from CPU to GPU
		cudaSafeCall(cudaMemcpy(dp_Buffer_Images[c], dataset.data() + channelOffset, elemCountTotal * sizeof(float), cudaMemcpyHostToDevice));
	}

	uint huffmanBits = 0;
	// Instantiate helper classes for managing shared GPU resources, maybe overkill
	GPUResources::Config config = CompressVolumeResources::getRequiredResources(DimX, DimY, DimZ, (uint)channelCount, huffmanBits);
	GPUResources shared;
	shared.create(config);
	CompressVolumeResources res;
	res.create(shared.getConfig());

	std::vector<std::vector<uint>> bitStreams(channelCount);	// [?] Bitstream buffers per channel

	//printf("Press Enter to continue... (2) \n");
	//getchar();

	// Compress!
	for (uint i = 0; i < iterations; i++)
	{
		for (size_t c = 0; c < channelCount; c++)
		{
			compressVolumeFloat(shared, res, dp_Buffer_Images[c], DimX, DimY, DimZ, numLevels, bitStreams[c], quantStep, doRLEOnlyOnLvl0);
		}
	}

	// Wait for completion
	cudaDeviceSynchronize();

	//printf("Press Enter to continue... (3) \n");
	//getchar();

	// Reset memory to 0
	for (size_t c = 0; c < channelCount; c++)
	{
		cudaSafeCall(cudaMemset(dp_Buffer_Images[c], 0, elemCountTotal * sizeof(float)));
	}

	//printf("Press Enter to continue... (4) \n");
	//getchar();

	// Register bitstreams again for use by device (this may here not be necessary as we have not used bitStreams to save data on the Host)
	for (size_t c = 0; c < channelCount; c++)
	{
		cudaSafeCall(cudaHostRegister(bitStreams[c].data(), bitStreams[c].size() * sizeof(uint), cudaHostRegisterDefault));
	}

	// Prepare decompression
	std::vector<VolumeChannel> channels(channelCount);
	for (size_t c = 0; c < channelCount; c++)
	{
		channels[c].dpImage = dp_Buffer_Images[c];
		channels[c].pBits = bitStreams[c].data();
		channels[c].bitCount = uint(bitStreams[c].size() * sizeof(uint) * 8);
		channels[c].quantizationStepLevel0 = quantStep;
	}

	//printf("Press Enter to continue... (5) \n");
	//getchar();

	// Decompress!
	for (uint i = 0; i < iterations; i++)
	{
		//decompressVolumeFloatMultiChannel(shared, res, channels.data(), (uint)channels.size(), DimX, DimY, DimZ, numLevels, doRLEOnlyOnLvl0);
		decompressVolumeFloatMultiChannel(shared, res, channels.data(), (uint)channels.size(), DimX, DimY, DimZ, numLevels, doRLEOnlyOnLvl0);
	}

	//printf("Press Enter to continue... (6) \n");
	//getchar();

	// Unregister bitstream to make it pageable again
	for (size_t c = 0; c < channelCount; c++) {
		cudaSafeCall(cudaHostUnregister(bitStreams[c].data()));
	}

	// Wait for completion
	cudaDeviceSynchronize();

	//printf("Press Enter to continue... (8) \n");
	//getchar();

	// Copy data to host
	for (size_t c = 0; c < channelCount; c++) {
		const auto channelOffset = c * DimX * DimY * DimZ * DimT;
		// First reset any old data with 0
		memset(compressed_dataset.data() + channelOffset, 0, elemCountTotal * sizeof(float));
		// Then copy from device to host
		cudaSafeCall(cudaMemcpy(compressed_dataset.data() + channelOffset, dp_Buffer_Images[c], elemCountTotal * sizeof(float), cudaMemcpyDeviceToHost));
	}

	// Deinstantiate compression related objects
	res.destroy();
	shared.destroy();
	for (size_t c = 0; c < channelCount; c++) {
		cudaSafeCall(cudaFree(dp_Buffer_Images[c]));
	}

	printf("Dataset Size:  (%i, %i, %i) [(x, y, z)]\n", DimX, DimY, DimZ);

	// Report compression efficiency
	uint BytesCompressed = 0;
	for (size_t c = 0; c < channelCount; c++) {
		BytesCompressed += uint(bitStreams[c].size()) * sizeof(uint);
	}
	float compressionFactor = sizeof(float) * float(elemCountTotal) * float(channelCount) / float(BytesCompressed);
	printf("Compressed size: %u B  (%.2f : 1)\n", BytesCompressed, compressionFactor);

	double error_x_avg = 0.0;
	double error_y_avg = 0.0;
	double error_z_avg = 0.0;
	for (auto i = 0; i < dataset.size() / 3; i++)
	{
		const auto channelOffset = DimX * DimY * DimZ * DimT;

		error_x_avg += abs(dataset[i + 0 * channelOffset] - compressed_dataset[i + 0 * channelOffset]);
		error_y_avg += abs(dataset[i + 1 * channelOffset] - compressed_dataset[i + 1 * channelOffset]);
		error_z_avg += abs(dataset[i + 2 * channelOffset] - compressed_dataset[i + 2 * channelOffset]);
	}

	error_x_avg /= (dataset.size() / 3);
	error_y_avg /= (dataset.size() / 3);
	error_z_avg /= (dataset.size() / 3);

	printf("Compression Error:  (%.4f, %.4f, %.4f) [(x, y, z)]\n", error_x_avg, error_y_avg, error_z_avg);

	double rms_dist = 0.0;
	for (auto i = 0; i < dataset.size() / 3; i++)
	{
		const auto channelOffset = DimX * DimY * DimZ * DimT;
		auto src = glm::vec3(dataset[i + 0 * channelOffset], dataset[i + 1 * channelOffset], dataset[i + 2 * channelOffset]);
		auto enc = glm::vec3(compressed_dataset[i + 0 * channelOffset], compressed_dataset[i + 1 * channelOffset], compressed_dataset[i + 2 * channelOffset]);

		rms_dist += glm::distance(src, enc) * glm::distance(src, enc);
	}

	rms_dist = glm::sqrt(rms_dist / (dataset.size() / 3));
	printf("Compression RMSE:  %.4f\n", rms_dist);

	return 1;
}
*/

int TestCudaCompress()
{
	TestCudaCompressKernel();
	return 1;
}

int TestCuda()
{
	TestCUDAKernel();
	return 1;
}

int TestParticleTracer()
{
	TestParticleKernel();
	return 1;
}

int TestTexture()
{
	TestTextureKernel();
	return 1;
}

int TestCompressedParticleTracer()
{
	TestCompressedParticleKernel();
	return 1;
}