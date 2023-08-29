#include "cudaFuncs.cuh"

#include <stdio.h>
#include <cstdint>
#include <chrono>
#include <array>
#include <cstddef>
#include <string_view>

#include <cuda_runtime.h>

#include "CompressVolume.h"
#include "Datasets.h"
#include "ParticleHelpers.h"

#include "cudaHelpers.cuh"
#include "cudaKernels.cui"
#include "half.h"

#include <fstream>
#include <spdlog/fmt/bundled/core.h>
#include <spdlog/fmt/bundled/ostream.h>
#include <filesystem>


void PrintDeviceInformation()
{
	// Device Information
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	printf("System Info (GPU #0)\n");
	printf("--------------------\n");
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	printf("  Name: %s\n", props.name);
	printf("  Compute Capability: %i.%i\n", props.major, props.minor);
	printf("  Warp Size: %i\n", props.warpSize);
	printf("  Available Memory: %.2f GB\n", props.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
	printf("  Memory Bus Width: %i Bit\n", props.memoryBusWidth);
	printf("  Peak Memory Bandwidth: %.2f GB/s\n\n", 2.0 * props.memoryClockRate * (props.memoryBusWidth / 8) / 1.0e6);

	printf("Execute\n");
	printf("-------\n");
}

template <typename T>
std::array<double, 3> avg_error(T* data_ptr1, T* data_ptr2, size_t numVecs, int channelCount, size_t channelStride = 1)
{
	double error_x_avg = 0.0;
	double error_y_avg = 0.0;
	double error_z_avg = 0.0;

	for (auto i = 0; i < numVecs; i++)
	{
		if (channelCount > 0)
		    error_x_avg += abs(data_ptr1[i + 0U * channelStride] - data_ptr2[i + 0U * channelStride]);
		if (channelCount > 1)
		    error_y_avg += abs(data_ptr1[i + 1U * channelStride] - data_ptr2[i + 1U * channelStride]);
		if (channelCount > 2)
		    error_z_avg += abs(data_ptr1[i + 2U * channelStride] - data_ptr2[i + 2U * channelStride]);
	}

	error_x_avg /= numVecs;
	error_y_avg /= numVecs;
	error_z_avg /= numVecs;

	printf("Average Error: (%.4f, %.4f, %.4f) [(x, y, z)]\n", error_x_avg, error_y_avg, error_z_avg);

	return { error_x_avg, error_y_avg, error_z_avg };
}

float CompressionEfficiency(const std::vector<std::vector<uint>>& compressedBitStreams, size_t numVecs, int channelCount)
{
	// Report compression efficiency
	uint BytesCompressed = 0;
	for (size_t c = 0; c < channelCount; c++) {
		BytesCompressed += uint(compressedBitStreams[c].size()) * sizeof(uint);
	}
	float compressionFactor = sizeof(float) * (float)numVecs * (float)channelCount / float(BytesCompressed);
	printf("Compressed size: %u B (%.2f:1)\n", BytesCompressed, compressionFactor);

	return compressionFactor;
}

float CompressionEfficiency(const std::vector<std::vector<std::vector<uint>>>& compressedBitStreamBlocks, size_t numVecs, int channelCount)
{
	uint64_t BytesCompressed = 0;

	for (const auto& block : compressedBitStreamBlocks)
	{
	    // Report compression efficiency
	    for (size_t c = 0; c < channelCount; c++) {
		    BytesCompressed += uint(block[c].size()) * sizeof(uint);
	    }
	}
	float compressionFactor = sizeof(float) * (double)numVecs * (double)channelCount / double(BytesCompressed);
	printf("Compressed size: %u B (%.2f:1)\n", BytesCompressed, compressionFactor);

	return compressionFactor;
}

bool SaveCompressedVectorfield(const CompVectorfield& vf, const char* filepath)
{
	const auto absolute_dataset_path = std::filesystem::absolute(filepath);
	const auto dataset_filename = absolute_dataset_path.filename().string();
	std::time_t t = std::time(0); // get time now
	std::tm* now = std::localtime(&t);
	std::array<std::string_view, 7> weekdays = {
		"Sunday",
		"Monday",
		"Tuesday",
		"Wednesday",
		"Thursday",
		"Friday",
		"Saturday",
	};

	
	std::fstream file_comp;
	const std::string save_path = fmt::format("{}/{}_cudaComp_compressed_{}_{}_{}_{}.cudaComp",
		std::filesystem::path(filepath).parent_path().has_parent_path() ? std::filesystem::path(filepath).parent_path().string() : ".",
		absolute_dataset_path.filename().replace_extension("").string(),
		vf.numDecompositionLevels,
		vf.quantizationStepSize,
		vf.compressIterations,
		vf.huffmanBits
	);

	printf("Saving compression info...\n");
	file_comp.open(save_path, std::ios::out | std::ios::binary);
	file_comp.write(reinterpret_cast<const char*>(&vf.Dimensions), sizeof(vf.Dimensions));
	file_comp.write(reinterpret_cast<const char*>(&vf.ChannelCount), sizeof(vf.ChannelCount));
	file_comp.write(reinterpret_cast<const char*>(&vf.numDecompositionLevels), sizeof(vf.numDecompositionLevels));
	file_comp.write(reinterpret_cast<const char*>(&vf.quantizationStepSize), sizeof(vf.quantizationStepSize));
	file_comp.write(reinterpret_cast<const char*>(&vf.compressIterations), sizeof(vf.compressIterations));
	file_comp.write(reinterpret_cast<const char*>(&vf.huffmanBits), sizeof(vf.huffmanBits));
	printf("Saving compressed data...\n");
	for (uint w = 0; w < vf.Dimensions.w; w++)
	{
		for (uint c = 0; c < vf.ChannelCount; c++)
		{
			size_t compressedBytes = vf.hData[w][c].size() * sizeof(uint);
			file_comp.write(reinterpret_cast<const char*>(&compressedBytes), sizeof(size_t));
			file_comp.write(reinterpret_cast<const char*>(vf.hData[w][c].data()), compressedBytes);
		}
	}
	file_comp.close();

	return 0;
}


template<typename T>
double Launch1Texture4CTest(const Vectorfield<T>& vf, dim3 numBlocks, dim3 threadsPerBlock)
{
	TextureManager texManager{};
	texManager.RegisterTexture(vf.Data.data(), vf.Dimensions, vf.ChannelCount);

	// Recording Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaSafeCall(cudaDeviceSynchronize());


	cudaEventRecord(start);
	testTextureVector << <numBlocks, threadsPerBlock >> > (texManager.GetTextureObject(0), vf.Dimensions);
	cudaEventRecord(stop);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaEventSynchronize(stop);

	// Measure
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return milliseconds;
}

template <typename T>
double Launch3Texture1CTest(const Vectorfield<T>& vf, dim3 numBlocks, dim3 threadsPerBlock)
{
	TextureManager texManager{};
	auto elems4D = (size_t)vf.Dimensions.x * (size_t)vf.Dimensions.y * (size_t)vf.Dimensions.z * (size_t)vf.Dimensions.w;
	texManager.RegisterTexture(vf.Data.data(), vf.Dimensions, 1);
	texManager.RegisterTexture(vf.Data.data() + elems4D, vf.Dimensions, 1);
	texManager.RegisterTexture(vf.Data.data() + 2 * elems4D, vf.Dimensions, 1);

	// Recording Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel Exec
	cudaSafeCall(cudaDeviceSynchronize());

	cudaEventRecord(start);
	testTextureScalar << <numBlocks, threadsPerBlock >> > (texManager.GetTextureObject(0), texManager.GetTextureObject(1), texManager.GetTextureObject(2), vf.Dimensions);
	cudaEventRecord(stop);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaEventSynchronize(stop);

	// Measure
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return milliseconds;
}

template <typename T>
double Launch1BufferTest(const Vectorfield<T>& vf, dim3 numBlocks, dim3 threadsPerBlock)
{
	// Data Upload
	float4* d_dataset = nullptr;
	CreateDevicePointer(&d_dataset, vf.Data.size() * sizeof(float));
	cudaMemcpy(d_dataset, vf.Data.data(), vf.Data.size() * sizeof(float), cudaMemcpyHostToDevice);

	// Recording Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel Exec
	cudaSafeCall(cudaDeviceSynchronize());

	cudaEventRecord(start);
	testBufferVector << <numBlocks, threadsPerBlock >> > (d_dataset, vf.Dimensions);
	cudaEventRecord(stop);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaEventSynchronize(stop);

	// Measure
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_dataset);

	return milliseconds;
}

template <typename T>
double Launch3BufferTest(const Vectorfield<T>& vf, dim3 numBlocks, dim3 threadsPerBlock)
{
	auto elems4D = (size_t)vf.Dimensions.x * (size_t)vf.Dimensions.y * (size_t)vf.Dimensions.z * (size_t)vf.Dimensions.w;

	// Data Upload
	float* d_dataset_x = nullptr;
	CreateDevicePointer(&d_dataset_x, elems4D * sizeof(float));
	cudaMemcpy(d_dataset_x, vf.Data.data(), elems4D * sizeof(float), cudaMemcpyHostToDevice);


	float* d_dataset_y = nullptr;
	CreateDevicePointer(&d_dataset_y, elems4D * sizeof(float));
	cudaMemcpy(d_dataset_y, vf.Data.data() + elems4D, elems4D * sizeof(float), cudaMemcpyHostToDevice);

	float* d_dataset_z = nullptr;
	CreateDevicePointer(&d_dataset_z, elems4D * sizeof(float));
	cudaMemcpy(d_dataset_z, vf.Data.data() + 2 * elems4D, elems4D * sizeof(float), cudaMemcpyHostToDevice);

	// Recording Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel Exec
	cudaSafeCall(cudaDeviceSynchronize());

	cudaEventRecord(start);
	testBufferScalar << <numBlocks, threadsPerBlock >> > (d_dataset_x, d_dataset_y, d_dataset_z, vf.Dimensions);
	cudaEventRecord(stop);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaEventSynchronize(stop);

	// Measure
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(d_dataset_x);
	cudaFree(d_dataset_y);
	cudaFree(d_dataset_z);

	return milliseconds;
}

template <typename T>
double Launch1Texture4CIntegrationTest(const Vectorfield<T>& vf, const IntegrationConfig& conf, dim3 numBlocks, dim3 threadsPerBlock, int iterations = 2)
{
	TextureManager texManager{};
	auto elems3D = (size_t)vf.Dimensions.z * (size_t)vf.Dimensions.y * (size_t)vf.Dimensions.x;

	for (size_t tidx = 0; tidx < vf.Dimensions.w; tidx++)
	{
		texManager.RegisterTexture(vf.Data.data() + tidx * elems3D * (size_t)vf.ChannelCount, vf.Dimensions, vf.ChannelCount);
	}

	size_t maxNumPoints = conf.SeedsNum() * (conf.Steps + 1);

	// Device Buffers
	float4* d_OutPosBuffer = nullptr;
	float4* d_OutVeloBuffer = nullptr;
	uint32_t* d_TraceLengthBuffer = nullptr;

	CreateDevicePointer(&d_OutPosBuffer, maxNumPoints * sizeof(float4));
	CreateDevicePointer(&d_OutVeloBuffer, maxNumPoints * sizeof(float4));
	CreateDevicePointer(&d_TraceLengthBuffer, conf.SeedsNum() * sizeof(uint32_t));

	// Recording Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel Execution
	cudaEventRecord(start);

	for (auto i = 0; i < iterations; i++)
	{
		seedParticles << <numBlocks, threadsPerBlock >> > (texManager.GetDeviceTextureObjects(), vf.Dimensions, conf, d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);
		traceParticles << <numBlocks, threadsPerBlock >> > (texManager.GetDeviceTextureObjects(), vf.Dimensions, conf, d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);
	}

	cudaEventRecord(stop);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaSafeCall(cudaEventSynchronize(stop));

	// Measure
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	auto time = milliseconds / (double)iterations;
	printf("1 Texture time: %.6f ms\n", time);

	// Device to Host copy
	std::vector<float4> ParticlePositions(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<float4> ParticleVelocities(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<uint32_t> ParticleTraceLength(conf.SeedsNum());

	cudaSafeCall(cudaMemcpy(ParticlePositions.data(), d_OutPosBuffer, ParticlePositions.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleVelocities.data(), d_OutVeloBuffer, ParticleVelocities.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleTraceLength.data(), d_TraceLengthBuffer, ParticleTraceLength.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaDeviceSynchronize());

	// Read
	float4 ParticleSum = make_float4(0.0f);
	for (auto i = 0; i < ParticleTraceLength[0]; i++)
	{
		ParticleSum += ParticlePositions[i];
	}

	printf("Particle 0 Sum: (%.2f, %.2f, %.2f)\n\n", ParticleSum.x, ParticleSum.y, ParticleSum.z);

	// Destroy Buffers
	cudaSafeCall(cudaFree(d_OutPosBuffer));
	cudaSafeCall(cudaFree(d_OutVeloBuffer));
	cudaSafeCall(cudaFree(d_TraceLengthBuffer));

	return time;
}

template <typename T>
std::vector<double> Launch3Texture1CIntegrationTest(const Vectorfield<T>& vf, const IntegrationConfig& conf, dim3 numBlocks, dim3 threadsPerBlock, int iterations = 2)
{
	TextureManager texManager{};

	auto elems3D = (size_t)vf.Dimensions.x * (size_t)vf.Dimensions.y * (size_t)vf.Dimensions.z;
	for (size_t tidx = 0; tidx < vf.Dimensions.w * vf.ChannelCount; tidx++)
	{
		texManager.RegisterTexture(vf.Data.data() + tidx * elems3D, vf.Dimensions, 1);
	}

	// Device Buffers
	float4* d_OutPosBuffer = nullptr;
	float4* d_OutVeloBuffer = nullptr;
	uint32_t* d_TraceLengthBuffer = nullptr;

	size_t maxNumPoints = conf.SeedsNum() * (conf.Steps + 1);
	CreateDevicePointer(&d_OutPosBuffer, maxNumPoints * sizeof(float4));
	CreateDevicePointer(&d_OutVeloBuffer, maxNumPoints * sizeof(float4));
	CreateDevicePointer(&d_TraceLengthBuffer, conf.SeedsNum() * sizeof(uint32_t));

	auto d_textures3C = texManager.GetDeviceTextureObjects3C();

	// Recording Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel Execution
	
	std::vector<double> times;
    for (auto i = 0; i < iterations; i++)
	{
		cudaEventRecord(start);
		seedParticles << <numBlocks, threadsPerBlock >> > (
			d_textures3C,
			vf.Dimensions, conf,
			d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);
		traceParticles << <numBlocks, threadsPerBlock >> > (
			d_textures3C,
			vf.Dimensions, conf,
			d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);
		cudaEventRecord(stop);

		cudaSafeCall(cudaEventSynchronize(stop));
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		times.push_back(milliseconds);
	}

	cudaSafeCall(cudaDeviceSynchronize());

	//// Measure
	//float milliseconds = 0;
	//cudaEventElapsedTime(&milliseconds, start, stop);
	//cudaEventDestroy(start);
	//cudaEventDestroy(stop);
	//
	//auto time = milliseconds / (double)iterations;
	//printf("3 Textures time: %.6f ms\n", time);

	// Device to Host copy
	std::vector<float4> ParticlePositions(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<float4> ParticleVelocities(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<uint32_t> ParticleTraceLength(conf.SeedsNum());

	cudaSafeCall(cudaMemcpy(ParticlePositions.data(), d_OutPosBuffer, ParticlePositions.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleVelocities.data(), d_OutVeloBuffer, ParticleVelocities.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleTraceLength.data(), d_TraceLengthBuffer, ParticleTraceLength.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaDeviceSynchronize());

	// Read
	float4 ParticleSum = make_float4(0.0f);
	for (auto i = 0; i < ParticleTraceLength[0]; i++)
	{
		ParticleSum += ParticlePositions[i];
	}

	printf("Particle 0 Sum: (%.2f, %.2f, %.2f)\n\n", ParticleSum.x, ParticleSum.y, ParticleSum.z);

	// Destroy Buffers
	cudaSafeCall(cudaFree(d_OutPosBuffer));
	cudaSafeCall(cudaFree(d_OutVeloBuffer));
	cudaSafeCall(cudaFree(d_TraceLengthBuffer));

	return times;
}

template <typename T>
float Launch1BufferIntegrationTest(const Vectorfield<T>& vf, const IntegrationConfig& conf, dim3 numBlocks, dim3 threadsPerBlock, int iterations = 2)
{
	// Data Upload
	float4* d_dataset = nullptr;
	CreateDevicePointer(&d_dataset, vf.Data.size() * sizeof(float));
	cudaSafeCall(cudaMemcpy(d_dataset, vf.Data.data(), vf.Data.size() * sizeof(float), cudaMemcpyHostToDevice));

	// Device Buffers
	float4* d_OutPosBuffer = nullptr;
	float4* d_OutVeloBuffer = nullptr;
	uint32_t* d_TraceLengthBuffer = nullptr;

	size_t maxNumPoints = conf.SeedsNum() * (conf.Steps + 1);
	CreateDevicePointer(&d_OutPosBuffer, maxNumPoints * sizeof(float4));
	CreateDevicePointer(&d_OutVeloBuffer, maxNumPoints * sizeof(float4));
	CreateDevicePointer(&d_TraceLengthBuffer, conf.SeedsNum() * sizeof(uint32_t));

	// Recording Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel Exec
	cudaSafeCall(cudaDeviceSynchronize());

	cudaEventRecord(start);

	for (auto i = 0; i < iterations; i++)
	{
		seedParticles << <numBlocks, threadsPerBlock >> > (
			d_dataset,
			vf.Dimensions, conf,
			d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);
		
		traceParticles << <numBlocks, threadsPerBlock >> > (
			d_dataset,
			vf.Dimensions, conf,
			d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);
	}

	cudaEventRecord(stop);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaEventSynchronize(stop);

	// Measure
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	auto time = milliseconds / (double)iterations;
	printf("1 Buffer time: %.6f ms\n", time);

	// Device to Host copy
	std::vector<float4> ParticlePositions(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<float4> ParticleVelocities(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<uint32_t> ParticleTraceLength(conf.SeedsNum());

	cudaSafeCall(cudaMemcpy(ParticlePositions.data(), d_OutPosBuffer, ParticlePositions.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleVelocities.data(), d_OutVeloBuffer, ParticleVelocities.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleTraceLength.data(), d_TraceLengthBuffer, ParticleTraceLength.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaDeviceSynchronize());

	// Read
	float4 ParticleSum = make_float4(0.0f);
	for (auto i = 0; i < ParticleTraceLength[0]; i++)
	{
		ParticleSum += ParticlePositions[i];
	}

	printf("Particle 0 Sum: (%.2f, %.2f, %.2f)\n\n", ParticleSum.x, ParticleSum.y, ParticleSum.z);

	// Destroy Buffers
	cudaSafeCall(cudaFree(d_dataset));
	cudaSafeCall(cudaFree(d_OutPosBuffer));
	cudaSafeCall(cudaFree(d_OutVeloBuffer));
	cudaSafeCall(cudaFree(d_TraceLengthBuffer));

	return time;
}

template <typename T>
double Launch3BufferIntegrationTest(const Vectorfield<T>& vf, const IntegrationConfig& conf, dim3 numBlocks, dim3 threadsPerBlock, int iterations = 2)
{
	Buffers3C d_buffers{};

	auto elems4D = (size_t)vf.Dimensions.x * (size_t)vf.Dimensions.y * (size_t)vf.Dimensions.z * (size_t)vf.Dimensions.w;

	// Data Upload
	CreateDevicePointer(&d_buffers.x, vf.Data.size() * sizeof(float));
	CreateDevicePointer(&d_buffers.y, vf.Data.size() * sizeof(float));
	CreateDevicePointer(&d_buffers.z, vf.Data.size() * sizeof(float));
	cudaSafeCall(cudaMemcpy(d_buffers.x, vf.Data.data(), elems4D * sizeof(float), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(d_buffers.y, vf.Data.data() + 1 * elems4D, elems4D * sizeof(float), cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMemcpy(d_buffers.z, vf.Data.data() + 2 * elems4D, elems4D * sizeof(float), cudaMemcpyHostToDevice));

	// Device Buffers
	float4* d_OutPosBuffer = nullptr;
	float4* d_OutVeloBuffer = nullptr;
	uint32_t* d_TraceLengthBuffer = nullptr;

	size_t maxNumPoints = conf.SeedsNum() * (conf.Steps + 1);
	CreateDevicePointer(&d_OutPosBuffer, maxNumPoints * sizeof(float4));
	CreateDevicePointer(&d_OutVeloBuffer, maxNumPoints * sizeof(float4));
	CreateDevicePointer(&d_TraceLengthBuffer, conf.SeedsNum() * sizeof(uint32_t));

	// Recording Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel Exec
	cudaSafeCall(cudaDeviceSynchronize());

	cudaEventRecord(start);

	for (auto i = 0; i < iterations; i++)
	{
		seedParticles << <numBlocks, threadsPerBlock >> > (
			d_buffers,
			vf.Dimensions, conf,
			d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);

		traceParticles << <numBlocks, threadsPerBlock >> > (
			d_buffers,
			vf.Dimensions, conf,
			d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);
	}

	cudaEventRecord(stop);

	cudaSafeCall(cudaDeviceSynchronize());
	cudaEventSynchronize(stop);

	// Measure
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	auto time = milliseconds / (double)iterations;
	printf("3 Buffer time: %.6f ms\n", time);

	// Device to Host copy
	std::vector<float4> ParticlePositions(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<float4> ParticleVelocities(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<uint32_t> ParticleTraceLength(conf.SeedsNum());

	cudaSafeCall(cudaMemcpy(ParticlePositions.data(), d_OutPosBuffer, ParticlePositions.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleVelocities.data(), d_OutVeloBuffer, ParticleVelocities.size() * sizeof(float4), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleTraceLength.data(), d_TraceLengthBuffer, ParticleTraceLength.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaDeviceSynchronize());

	// Read
	float4 ParticleSum = make_float4(0.0f);
	for (auto i = 0; i < ParticleTraceLength[0]; i++)
	{
		ParticleSum += ParticlePositions[i];
	}

	printf("Particle 0 Sum: (%.2f, %.2f, %.2f)\n\n", ParticleSum.x, ParticleSum.y, ParticleSum.z);

	// Destroy Buffers
	cudaSafeCall(cudaFree(d_buffers.x));
	cudaSafeCall(cudaFree(d_buffers.y));
	cudaSafeCall(cudaFree(d_buffers.z));

	cudaSafeCall(cudaFree(d_OutPosBuffer));
	cudaSafeCall(cudaFree(d_OutVeloBuffer));
	cudaSafeCall(cudaFree(d_TraceLengthBuffer));

	return time;
}

void TestCUDAKernel()
{
	PrintDeviceInformation();

	dim3 threadsPerBlock(2, 2, 1);
	dim3 numBlocks(2, 1, 1);

	printf("Threads per Block: (%i, %i, %i)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
	printf("Number of Blocks: (%i, %i, %i)\n", numBlocks.x, numBlocks.y, numBlocks.z);
	printf("Launching kernel...\n");
	kernelDimensionTest << <numBlocks, threadsPerBlock >> > ();
}

void TestTextureKernel()
{
	PrintDeviceInformation();

	uint4 Dimensions{ 256 , 256, 256, 1 };

	Vectorfield<float> vF_4CInterleaved{};
	vF_4CInterleaved.Dimensions = Dimensions;
	vF_4CInterleaved.ChannelCount = 4;
	vF_4CInterleaved.InterleavedXYZ = true;
	vF_4CInterleaved.Data = GenerateTestDataset<float>(vF_4CInterleaved.Dimensions, vF_4CInterleaved.ChannelCount, vF_4CInterleaved.InterleavedXYZ);
	//vF_4CInterleaved.Data = GenerateABCDataset<float>(vF_4CInterleaved.Dimensions, vF_4CInterleaved.ChannelCount, vF_4CInterleaved.InterleavedXYZ);

	Vectorfield<float> vF_3C{};
	vF_3C.Dimensions = Dimensions;
	vF_3C.ChannelCount = 3;
	vF_3C.InterleavedXYZ = false;
	vF_3C.Data = GenerateTestDataset<float>(vF_3C.Dimensions, vF_3C.ChannelCount, vF_3C.InterleavedXYZ);
	//vF_3C.Data = GenerateABCDataset<float>(vF_3C.Dimensions, vF_3C.ChannelCount, vF_3C.InterleavedXYZ);

	dim3 threadsPerBlock(1);
	dim3 numBlocks(1);

	printf("Launching Tests\n");
	printf("Data Dimensinos: (%i, %i, %i, %i)\n", Dimensions.x, Dimensions.y, Dimensions.z, Dimensions.w);
	printf("Threads per Block:  (%i, %i, %i)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
	printf("Number of Blocks:   (%i, %i, %i)\n", numBlocks.x, numBlocks.y, numBlocks.z);
	printf("...\n\n");


	// CPU Test
	{
		float4* data_ptr = (float4*)vF_4CInterleaved.Data.data();
		float4 vector = make_float4(0.0);

		auto t_start = std::chrono::high_resolution_clock::now();
		uint elems2D = Dimensions.y * Dimensions.x;
		for (int z = 0; z < Dimensions.z; z++)
		{
			for (int y = 0; y < Dimensions.y; y++)
			{
			    for (int x = 0; x < Dimensions.x; x++)
				{
					//uint offset = z * elems2D + y * Dimensions.x + x;
					//vector += data_ptr[offset];
					vector += buf3D_CPU(data_ptr, Dimensions, x + 0.5f, y + 0.5f, z + 0.5f, 0.0f);
				}
			}
		}
		auto t_end = std::chrono::high_resolution_clock::now();
		auto time = std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count() / 1000.0;

		printf("CPU time:        %.6f ms\n", time);
	}

	// 1 Texture
	{
		auto time = Launch1Texture4CTest(vF_4CInterleaved, numBlocks, threadsPerBlock);
		printf("1 Texture time:  %.6f ms\n", time);
	}

	// 3 Textures
	{
		auto time = Launch3Texture1CTest(vF_3C, numBlocks, threadsPerBlock);
		printf("3 Textures time: %.6f ms\n", time);
	}

	// 1 Buffer
	{
		auto time = Launch1BufferTest(vF_4CInterleaved, numBlocks, threadsPerBlock);
		printf("1 Buffer time:   %.6f ms\n", time);
	}

	// 3 Buffers
	{
		auto time = Launch3BufferTest(vF_3C, numBlocks, threadsPerBlock);
		printf("3 Buffers time:  %.6f ms\n\n", time);
	} 
}

void TestParticleKernel()
{
	PrintDeviceInformation();
	uint4 Dimensions{ 256 , 256, 256, 10 };

	IntegrationConfig integrationConf{};
	integrationConf.Seeds = uint3{ 10, 12, 12 };
	integrationConf.CalculateStride(Dimensions);
	integrationConf.Steps = 1000;
	integrationConf.dt = 0.01f;

	Vectorfield<float> vF_4CInterleaved{};
	vF_4CInterleaved.Dimensions = Dimensions;
	vF_4CInterleaved.ChannelCount = 4;
	vF_4CInterleaved.InterleavedXYZ = true;
	//vF_4CInterleaved.Data = GenerateTestDataset<float>(vF_4CInterleaved.Dimensions, vF_4CInterleaved.ChannelCount, vF_4CInterleaved.InterleavedXYZ);
	vF_4CInterleaved.Data = GenerateABCDataset<float>(vF_4CInterleaved.Dimensions, vF_4CInterleaved.ChannelCount, vF_4CInterleaved.InterleavedXYZ);

	Vectorfield<float> vF_3C{};
	vF_3C.Dimensions = Dimensions;
	vF_3C.ChannelCount = 3;
	vF_3C.InterleavedXYZ = false;
	//vF_3C.Data = GenerateTestDataset<float>(vF_3C.Dimensions, vF_3C.ChannelCount, vF_3C.InterleavedXYZ);
	vF_3C.Data = GenerateABCDataset<float>(vF_3C.Dimensions, vF_3C.ChannelCount, vF_3C.InterleavedXYZ);

	Vectorfield<float> vF_3CInterleaved{};
	vF_3CInterleaved.Dimensions = Dimensions;
	vF_3CInterleaved.ChannelCount = 3;
	vF_3CInterleaved.InterleavedXYZ = true;
	//vF_3CInterleaved.Data = GenerateTestDataset<float>(vF_3CInterleaved.Dimensions, vF_3CInterleaved.ChannelCount, vF_3CInterleaved.InterleavedXYZ);
	vF_3CInterleaved.Data = GenerateABCDataset<float>(vF_3CInterleaved.Dimensions, vF_3CInterleaved.ChannelCount, vF_3CInterleaved.InterleavedXYZ);

	dim3 threadsPerBlock(2, 2, 2);
	dim3 numBlocks((integrationConf.Seeds.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(integrationConf.Seeds.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
		(integrationConf.Seeds.z + threadsPerBlock.z - 1) / threadsPerBlock.z);

	printf("Launching Tests\n");
	printf("Data Dimensinos: (%i, %i, %i, %i)\n", Dimensions.x, Dimensions.y, Dimensions.z, Dimensions.w);
	printf("Seeds:              (%i, %i, %i)\n", integrationConf.Seeds.x, integrationConf.Seeds.y, integrationConf.Seeds.z);
	printf("Steps:              %u\n", integrationConf.Steps);
	printf("Stepsize:           %.4f\n", integrationConf.dt);
	printf("Threads per Block:  (%i, %i, %i)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
	printf("Number of Blocks:   (%i, %i, %i)\n", numBlocks.x, numBlocks.y, numBlocks.z);
	printf("...\n\n");

	{
		traceParticles_CPU((float4*)vF_4CInterleaved.Data.data(), vF_4CInterleaved.Dimensions, integrationConf, make_uint3(0, 0, 0));
	}

	{
		Launch1Texture4CIntegrationTest(vF_4CInterleaved, integrationConf, numBlocks, threadsPerBlock);
	}

	{
		Launch3Texture1CIntegrationTest(vF_3C, integrationConf, numBlocks, threadsPerBlock);
	}

	{
		Launch1BufferIntegrationTest(vF_4CInterleaved, integrationConf, numBlocks, threadsPerBlock);
	}

	{
		Launch3BufferIntegrationTest(vF_3C, integrationConf, numBlocks, threadsPerBlock);
	}
}

template <typename T>
double compressWithCudaCompress(const Vectorfield<T>& src_vf, CompVectorfield& dst_vf, bool copyDevice = false)
{
	if (copyDevice)
	    dst_vf.dData.resize(src_vf.Dimensions.w);

	// Allocate GPU arrays to upload uncompressed data
	std::vector<float*> dp_Buffer_Images(src_vf.ChannelCount);
	for (size_t c = 0; c < src_vf.ChannelCount; c++)
	{
	    CreateDevicePointer(&dp_Buffer_Images[c], src_vf.NumVectors(3) * sizeof(float));
	}

	// Instantiate helper classes for managing shared GPU resources, maybe overkill
	GPUResources::Config GPUResourcesConfig = CompressVolumeResources::getRequiredResources(src_vf.Dimensions.x, src_vf.Dimensions.y, src_vf.Dimensions.z, (uint)src_vf.ChannelCount, dst_vf.huffmanBits);
	GPUResources GPUResources;
	GPUResources.create(GPUResourcesConfig);
	CompressVolumeResources CompVolumeResources;
	CompVolumeResources.create(GPUResources.getConfig());

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double gpu_time = 0.0;

	// Copy data to host
	for (auto t = 0; t < src_vf.Dimensions.w; t++)
	{
		for (size_t c = 0; c < src_vf.ChannelCount; c++)
		{
			cudaSafeCall(cudaMemset(dp_Buffer_Images[c], 0, src_vf.NumVectors(3) * sizeof(float)));
			const auto timeOffset = t * src_vf.NumVectors(3);
			const auto channelOffset = c * src_vf.NumVectors(4);
			cudaSafeCall(cudaMemcpy(dp_Buffer_Images[c], src_vf.Data.data() + timeOffset + channelOffset, src_vf.NumVectors(3) * sizeof(float), cudaMemcpyHostToDevice));
		}

	    dst_vf.hData[t].resize(src_vf.ChannelCount); // Bitstream buffers per channel
	    //std::vector<std::vector<uint>> bitStreams(vf.ChannelCount);	// Bitstream buffers per channel
	    // Compress!
		cudaEventRecord(start);
	    for (uint i = 0; i < dst_vf.compressIterations; i++)
	    {
		    for (size_t c = 0; c < src_vf.ChannelCount; c++)
		    {
			    compressVolumeFloat(GPUResources, CompVolumeResources, dp_Buffer_Images[c], src_vf.Dimensions.x, src_vf.Dimensions.y, src_vf.Dimensions.z, dst_vf.numDecompositionLevels, dst_vf.hData[t][c], dst_vf.quantizationStepSize, dst_vf.b_RLEOnlyOnLvl0);
		    }
	    }
		cudaEventRecord(stop);

		cudaSafeCall(cudaDeviceSynchronize());
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		gpu_time += milliseconds;
	}

	CompVolumeResources.destroy();
	GPUResources.destroy();

	for (size_t c = 0; c < src_vf.ChannelCount; c++)
	{
	    cudaSafeCall(cudaFree(dp_Buffer_Images[c]));
	}

	dst_vf.Dimensions = src_vf.Dimensions;
	dst_vf.ChannelCount = src_vf.ChannelCount;
	dst_vf.InterleavedXYZ = src_vf.InterleavedXYZ;

	return gpu_time;
}


template <typename T>
double compressWithCudaCompress(const char* filepath, CompVectorfield& dst_vf, bool interleaved)
{
	std::string dims_path = std::filesystem::path(filepath).replace_extension("").string() + "_dims.raw";
	uint4 Dimensions{ 0,0,0,0 };

	if (exists(std::filesystem::path(dims_path)))
	{
		std::fstream file;
		file.open(dims_path, std::ios::in | std::ios::binary);
		if (!file.good())
		{
			return -1;
		}
		file.read((char*)&Dimensions.x, sizeof(Dimensions));
		file.close();
	}

	// Open Dataset and read filesize and dimensions
	std::fstream file;
	file.open(filepath, std::ios::in | std::ios::binary | std::ios::ate);
	if (!file.good())
	{
		return -1.0;
	}
	auto filesize = file.tellg();
	file.seekg(0, std::ios::beg);

	bool seperate_dims = exists(std::filesystem::path(dims_path));
	if (!seperate_dims)
		file.read(reinterpret_cast<char*>(&Dimensions), sizeof(Dimensions));

	Vectorfield<float> src_vf;
	src_vf.ChannelCount = 3;
	src_vf.Dimensions = Dimensions;
	src_vf.InterleavedXYZ = false;
	src_vf.Data.resize(src_vf.NumScalars(3));

	// Allocate GPU arrays to upload uncompressed data
	std::vector<float*> dp_Buffer_Images(src_vf.ChannelCount);
	for (size_t c = 0; c < src_vf.ChannelCount; c++)
	{
		CreateDevicePointer(&dp_Buffer_Images[c], src_vf.NumVectors(3) * sizeof(float));
	}

	// Instantiate helper classes for managing shared GPU resources, maybe overkill
	GPUResources::Config GPUResourcesConfig = CompressVolumeResources::getRequiredResources(src_vf.Dimensions.x, src_vf.Dimensions.y, src_vf.Dimensions.z, (uint)src_vf.ChannelCount, dst_vf.huffmanBits);
	GPUResources GPUResources;
	GPUResources.create(GPUResourcesConfig);
	CompressVolumeResources CompVolumeResources;
	CompVolumeResources.create(GPUResources.getConfig());

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double gpu_time = 0.0;

	auto BytesChannel = src_vf.NumVectors(3) * sizeof(float);	// Bytes of a single channel
	for (auto t = 0; t < src_vf.Dimensions.w; t++)
	{
		float milliseconds = 0;

		//if (interleaved)
		if (false)
		{
			std::vector<T> tmp(src_vf.NumScalars(3));

			const auto fileTimeOffset = t * src_vf.NumScalars(3) * sizeof(float);

			file.seekg(sizeof(Dimensions) * !seperate_dims + fileTimeOffset, std::ios::beg);
			file.read(reinterpret_cast<char*>(tmp.data()), src_vf.NumScalars(3) * sizeof(float));

			for (auto i = 0; i < tmp.size() / src_vf.ChannelCount; i++)
			{
				for (auto c = 0; c < src_vf.ChannelCount; c++)
				{
					src_vf.Data[c * src_vf.NumVectors(3) + i] = tmp[c + i * src_vf.ChannelCount];
				}
			}

			for (auto c = 0; c < src_vf.ChannelCount; c++)
			{
				// Measure aux Buffers: necessary for cudaCompress
				cudaEventRecord(start);
				cudaSafeCall(cudaMemset(dp_Buffer_Images[c], 0, src_vf.NumVectors(3) * sizeof(float)));
				const auto channelOffset = c * src_vf.NumVectors(3);
				cudaSafeCall(cudaMemcpy(dp_Buffer_Images[c], src_vf.Data.data() + channelOffset, src_vf.NumVectors(3) * sizeof(float), cudaMemcpyHostToDevice));
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&milliseconds, start, stop);
				gpu_time += milliseconds;
			}
		}
		else {
			for (size_t c = 0; c < src_vf.ChannelCount; c++)
			{
				const auto fileTimeOffset = t * src_vf.NumVectors(3) * sizeof(float);
				const auto fileChannelOffset = c * src_vf.NumVectors(4) * sizeof(float);

				file.seekg(sizeof(Dimensions) * !seperate_dims + fileTimeOffset + fileChannelOffset, std::ios::beg);
				file.read(reinterpret_cast<char*>(&src_vf.Data[c * src_vf.NumVectors(3)]), BytesChannel);

				// Measure aux Buffers: necessary for cudaCompress
				cudaEventRecord(start);
				cudaSafeCall(cudaMemset(dp_Buffer_Images[c], 0, src_vf.NumVectors(3) * sizeof(float)));
				const auto channelOffset = c * src_vf.NumVectors(3);
				cudaSafeCall(cudaMemcpy(dp_Buffer_Images[c], src_vf.Data.data() + channelOffset, src_vf.NumVectors(3) * sizeof(float), cudaMemcpyHostToDevice));
				cudaEventRecord(stop);

				cudaEventSynchronize(stop);
				cudaEventElapsedTime(&milliseconds, start, stop);
				gpu_time += milliseconds;
			}
		}

		dst_vf.hData[t].resize(src_vf.ChannelCount); // Bitstream buffers per channel

		// Compress!
		// Measure Compression
		cudaEventRecord(start);
		for (uint i = 0; i < dst_vf.compressIterations; i++)
		{
			for (size_t c = 0; c < src_vf.ChannelCount; c++)
			{
				compressVolumeFloat(GPUResources, CompVolumeResources, dp_Buffer_Images[c], src_vf.Dimensions.x, src_vf.Dimensions.y, src_vf.Dimensions.z, dst_vf.numDecompositionLevels, dst_vf.hData[t][c], dst_vf.quantizationStepSize, dst_vf.b_RLEOnlyOnLvl0);
			}
		}
		cudaEventRecord(stop);

		cudaSafeCall(cudaDeviceSynchronize());
		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		gpu_time += milliseconds;
	}

	CompVolumeResources.destroy();
	GPUResources.destroy();

	for (size_t c = 0; c < src_vf.ChannelCount; c++)
	{
		cudaSafeCall(cudaFree(dp_Buffer_Images[c]));
	}

	dst_vf.Dimensions = src_vf.Dimensions;
	dst_vf.ChannelCount = src_vf.ChannelCount;
	dst_vf.InterleavedXYZ = src_vf.InterleavedXYZ;

	return gpu_time;
}

template <typename T>
double decompressWithCudaCompress(CompVectorfield& src_vf, Vectorfield<T>& dst_vf)
{
	// If dst vectorfield is pointer type, this means it holds device buffer
	//bool bDevice = std::is_pointer<T>::value;

	// Generate target buffers for decompression
	std::vector<float*> dp_Buffer_Images(src_vf.ChannelCount);
	for (auto c = 0; c < src_vf.ChannelCount > 0; c++)
	{
		CreateDevicePointer(&dp_Buffer_Images[c], src_vf.NumVectors(3) * sizeof(float));
	}

	// Instantiate helper classes for managing shared GPU resources, maybe overkill
	GPUResources::Config GPUResourcesConfig = CompressVolumeResources::getRequiredResources(src_vf.Dimensions.x, src_vf.Dimensions.y, src_vf.Dimensions.z, (uint)src_vf.ChannelCount, src_vf.huffmanBits);
	GPUResources GPUResources;
	GPUResources.create(GPUResourcesConfig);
	CompressVolumeResources CompVolumeResources;
	CompVolumeResources.create(GPUResources.getConfig());

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	double gpu_time = 0.0;

    for (auto t = 0; t < src_vf.Dimensions.w; t++)
    {
	    // Reset memory to 0
		cudaEventRecord(start);
	    for (size_t c = 0; c < src_vf.ChannelCount; c++)
	    {
		    cudaSafeCall(cudaMemset(dp_Buffer_Images[c], 0, src_vf.NumVectors(3) * sizeof(float)));
	    }
		cudaEventRecord(stop);

		cudaSafeCall(cudaDeviceSynchronize());
		cudaEventSynchronize(stop);
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		gpu_time += milliseconds;


	    // Register bitstreams for use by device
	    for (size_t c = 0; c < src_vf.ChannelCount; c++)
	    {
		    cudaSafeCall(cudaHostRegister(src_vf.hData[t][c].data(), src_vf.hData[t][c].size() * sizeof(uint), cudaHostRegisterDefault));
	    }

	    // Prepare decompression
	    std::vector<VolumeChannel> channels(src_vf.ChannelCount);
	    for (size_t c = 0; c < src_vf.ChannelCount; c++)
	    {
		    channels[c].dpImage = dp_Buffer_Images[c];
		    channels[c].pBits = src_vf.hData[t][c].data();
		    channels[c].bitCount = uint(src_vf.hData[t][c].size() * sizeof(uint) * 8);
		    channels[c].quantizationStepLevel0 = src_vf.quantizationStepSize;
	    }


	    // Decompress!
		cudaEventRecord(start);
	    for (uint i = 0; i < src_vf.compressIterations; i++)
	    {
		    decompressVolumeFloatMultiChannel(GPUResources, CompVolumeResources, channels.data(), (uint)channels.size(), src_vf.Dimensions.x, src_vf.Dimensions.y, src_vf.Dimensions.z, src_vf.numDecompositionLevels, src_vf.b_RLEOnlyOnLvl0);
	    }
		cudaEventRecord(stop);

		cudaSafeCall(cudaDeviceSynchronize());
		cudaEventSynchronize(stop);
		milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		gpu_time += milliseconds;


	    // Unregister bitstream to make it pageable again
	    for (size_t c = 0; c < src_vf.ChannelCount; c++) {
		    cudaSafeCall(cudaHostUnregister(src_vf.hData[t][c].data()));
	    }

	    // Wait for completion
		cudaSafeCall(cudaDeviceSynchronize());

	    // Copy data to host
		dst_vf.Data.resize(src_vf.NumScalars(4));
		const auto timeOffset = t * src_vf.NumVectors(3);
		for (size_t c = 0; c < src_vf.ChannelCount; c++) {
			const auto channelOffset = c * src_vf.NumVectors(4);

			// Copy from device to host
			cudaSafeCall(cudaMemcpy(dst_vf.Data.data() + timeOffset + channelOffset, dp_Buffer_Images[c], src_vf.NumVectors(3) * sizeof(float), cudaMemcpyDeviceToHost));
		}
	}

	// Deinstantiate compression related objects
	CompVolumeResources.destroy();
	GPUResources.destroy();

	for (auto c = 0; c < src_vf.ChannelCount > 0; c++)
	{
		cudaSafeCall(cudaFree(dp_Buffer_Images[c]));
	}
	
	dst_vf.Dimensions = src_vf.Dimensions;
	dst_vf.ChannelCount = src_vf.ChannelCount;
	dst_vf.InterleavedXYZ = src_vf.InterleavedXYZ;

	return gpu_time;
}

void TestCudaCompressKernel()
{
	PrintDeviceInformation();
	uint4 Dimensions{ 256 , 128, 128, 5 };

	Vectorfield<float> vf{};
	vf.Dimensions = Dimensions;
	vf.ChannelCount = 4;
	vf.InterleavedXYZ = false;
	//vf.Data = GenerateTestDataset<float>(vf.Dimensions, vf.ChannelCount, vf.InterleavedXYZ);
	vf.Data = { GenerateABCDataset<float>(vf.Dimensions, vf.ChannelCount, vf.InterleavedXYZ) };

	CompVectorfield c_vf{};
	c_vf.hData.resize(vf.Dimensions.w);
	c_vf.numDecompositionLevels = 2;		// Too many decompositions may introduce artifacts
	c_vf.quantizationStepSize = 0.00136f;	// Granularity of data precision reduction (impacts compression efficiency) def: 0.00136f
	c_vf.compressIterations = 10;			// May improve... something
	c_vf.huffmanBits = 0;
	c_vf.b_RLEOnlyOnLvl0 = true;

	printf("Launching Tests\n");
	printf("Decomposition Levels:   %i\n", c_vf.numDecompositionLevels);
	printf("Quantization Step Size: %.6f\n", c_vf.quantizationStepSize);
	printf("Compression Iterations: %u\n", c_vf.compressIterations);
	printf("Huffman Bits:           %u\n", c_vf.huffmanBits);
	printf("-----------------------------------------\n");
	printf("Data Dimensions:       (%i, %i, %i, %i)\n", Dimensions.x, Dimensions.y, Dimensions.z, Dimensions.w);
	printf("Data Size:             %.2f MB\n", vf.NumScalars() * sizeof(float) / 1000000.0);

	compressWithCudaCompress(vf, c_vf);

	// Wait for completion
	cudaDeviceSynchronize();

	Vectorfield<float> r_vf{};
    decompressWithCudaCompress(c_vf, r_vf);

	printf("Results:\n");
	CompressionEfficiency(c_vf.hData, vf.NumVectors(4), vf.ChannelCount);
	avg_error(vf.Data.data(), r_vf.Data.data(), vf.NumVectors(), vf.ChannelCount, (vf.InterleavedXYZ ? 1 : vf.NumVectors(4)));
}

void TestCompressedParticleKernel()
{
	PrintDeviceInformation();
	uint4 Dimensions{ 256 , 256, 256, 10 };

	IntegrationConfig integrationConf{};
	integrationConf.Seeds = uint3{ 10, 12, 12 };
	integrationConf.CalculateStride(Dimensions);
	integrationConf.Steps = 1000;
	integrationConf.dt = 0.01f;

	Vectorfield<float> vf{};
	vf.Dimensions = Dimensions;
	vf.ChannelCount = 3;
	vf.InterleavedXYZ = false;
	//vf.Data = GenerateTestDataset<float>(vf.Dimensions, vf.ChannelCount, vf.InterleavedXYZ);
	vf.Data = GenerateABCDataset<float>(vf.Dimensions, vf.ChannelCount, vf.InterleavedXYZ);

	CompVectorfield c_vf{};
	c_vf.hData.resize(vf.Dimensions.w);
	c_vf.numDecompositionLevels = 2;		// Too many decompositions may introduce artifacts
	c_vf.quantizationStepSize = 0.00136f;	// Granularity of data precision reduction (impacts compression efficiency) def: 0.00136f
	c_vf.compressIterations = 1;			// May improve... something
	c_vf.huffmanBits = 0;
	c_vf.b_RLEOnlyOnLvl0 = true;

	dim3 threadsPerBlock(2, 2, 2);
	dim3 numBlocks((integrationConf.Seeds.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(integrationConf.Seeds.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
		(integrationConf.Seeds.z + threadsPerBlock.z - 1) / threadsPerBlock.z);

	printf("Launching Tests\n");
	printf("Decomposition Levels:   %i\n", c_vf.numDecompositionLevels);
	printf("Quantization Step Size: %.6f\n", c_vf.quantizationStepSize);
	printf("Compression Iterations: %u\n", c_vf.compressIterations);
	printf("Huffman Bits:           %u\n", c_vf.huffmanBits);
	printf("-----------------------------------------\n");
	printf("Data Dimensions:        (%i, %i, %i, %i)\n", Dimensions.x, Dimensions.y, Dimensions.z, Dimensions.w);
	printf("Data Size:              %.2f MB\n", vf.NumScalars() * sizeof(float) / 1000000.0);
	printf("Seeds:                  (%i, %i, %i)\n", integrationConf.Seeds.x, integrationConf.Seeds.y, integrationConf.Seeds.z);
	printf("Steps:                  %u\n", integrationConf.Steps);
	printf("Stepsize:               %.4f\n", integrationConf.dt);
	printf("Threads per Block:      (%i, %i, %i)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
	printf("Number of Blocks:       (%i, %i, %i)\n", numBlocks.x, numBlocks.y, numBlocks.z);

	compressWithCudaCompress(vf, c_vf);

	// Wait for completion
	cudaDeviceSynchronize();

	Vectorfield<float> r_vf{};
	decompressWithCudaCompress(c_vf, r_vf);

	Launch3BufferIntegrationTest(vf, integrationConf, numBlocks, threadsPerBlock);
	Launch3BufferIntegrationTest(r_vf, integrationConf, numBlocks, threadsPerBlock);
	Launch3Texture1CIntegrationTest(r_vf, integrationConf, numBlocks, threadsPerBlock);

	printf("Results:\n");
	CompressionEfficiency(c_vf.hData, vf.NumVectors(4), vf.ChannelCount);
	avg_error(vf.Data.data(), r_vf.Data.data(), vf.NumVectors(), vf.ChannelCount, (vf.InterleavedXYZ ? 1 : vf.NumVectors(4)));
}

void TraceABC()
{
	PrintDeviceInformation();
	uint4 Dimensions{ 225 , 250, 200, 151 };

	Vectorfield<float> vf{};
	vf.Dimensions = Dimensions;
	vf.ChannelCount = 3;
	vf.InterleavedXYZ = false;
	vf.Data = { GenerateABCDataset<float>(vf.Dimensions, vf.ChannelCount, vf.InterleavedXYZ) };

	CompVectorfield vf_comp{};
	vf_comp.hData.resize(vf.Dimensions.w);
	vf_comp.numDecompositionLevels = 2;		// Too many decompositions may introduce artifacts
	vf_comp.quantizationStepSize = 0.00136f;	// Granularity of data precision reduction (impacts compression efficiency) def: 0.00136f
	vf_comp.compressIterations = 10;			// May improve... something
	vf_comp.huffmanBits = 0;
	vf_comp.b_RLEOnlyOnLvl0 = true;

	printf("Launching Tests\n");
	printf("Decomposition Levels:   %i\n", vf_comp.numDecompositionLevels);
	printf("Quantization Step Size: %.6f\n", vf_comp.quantizationStepSize);
	printf("Compression Iterations: %u\n", vf_comp.compressIterations);
	printf("Huffman Bits:           %u\n", vf_comp.huffmanBits);
	printf("-----------------------------------------\n");
	printf("Data Dimensions:       (%i, %i, %i, %i)\n", Dimensions.x, Dimensions.y, Dimensions.z, Dimensions.w);
	printf("Data Size:             %.2f MB\n", vf.NumScalars() * sizeof(float) / 1000000.0);

	// Compression
	auto t0 = std::chrono::high_resolution_clock::now();
	auto tgpu_compression = compressWithCudaCompress<float>(vf, vf_comp);
	auto t1 = std::chrono::high_resolution_clock::now();
	auto tcpu_compression = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	// Wait for completion
	cudaDeviceSynchronize();

	// Decompression
	t0 = std::chrono::high_resolution_clock::now();
	auto tgpu_decompression = decompressWithCudaCompress(vf_comp, vf);
	t1 = std::chrono::high_resolution_clock::now();
	auto tcpu_decompression = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	CompressionEfficiency(vf_comp.hData, vf.NumVectors(4), vf.ChannelCount);
	printf("\n");
	
	printf("Time (GPU) Compression:      %.4fms\n", tgpu_compression);
	printf("Time (GPU) Decompression:    %.4fms\n", tgpu_decompression);
	printf("Time (CPU) Compression:      %ums\n", tcpu_compression);
	printf("Time (CPU) Decompression:    %ums\n", tcpu_decompression);
}


void TraceFile(const char* filepath, bool read_slicewise)
{
	PrintDeviceInformation();

	// Read dimensions of processed dataset
	uint4 Dimensions{ 0, 0, 0, 0 };
	std::fstream file;
	file.open(filepath, std::ios::in | std::ios::binary | std::ios::ate);
	if (!file.good())
	{
		return;
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
	auto t0 = std::chrono::high_resolution_clock::now();
	file.read(reinterpret_cast<char*>(vf_src.Data.data()), filesize - sizeof(Dimensions));
	auto t1 = std::chrono::high_resolution_clock::now();
	auto tcpu_fileread = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	CompVectorfield vf_comp{};
	vf_comp.hData.resize(vf_src.Dimensions.w);
	vf_comp.numDecompositionLevels = 2;		// Too many decompositions may introduce artifacts
	vf_comp.quantizationStepSize = 0.00136f;	// Granularity of data precision reduction (impacts compression efficiency) def: 0.00136f
	vf_comp.compressIterations = 10;			// May improve... something
	vf_comp.huffmanBits = 0;
	vf_comp.b_RLEOnlyOnLvl0 = true;

	printf("Launching Compression\n");
	printf("Decomposition Levels:   %i\n", vf_comp.numDecompositionLevels);
	printf("Quantization Step Size: %.6f\n", vf_comp.quantizationStepSize);
	printf("Compression Iterations: %u\n", vf_comp.compressIterations);
	printf("Huffman Bits:           %u\n", vf_comp.huffmanBits);
	printf("-----------------------------------------\n");
	printf("Data Dimensions:       (%i, %i, %i, %i)\n", Dimensions.x, Dimensions.y, Dimensions.z, Dimensions.w);
	printf("Data Size:             %.2f MB\n", vf_src.NumScalars() * sizeof(float) / 1000000.0);
	printf("...\n");

	// Compression
	t0 = std::chrono::high_resolution_clock::now();
	auto tgpu_compression = compressWithCudaCompress<float>(filepath, vf_comp, false);
	t1 = std::chrono::high_resolution_clock::now();
	auto tcpu_compression = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	// Wait for completion
	cudaDeviceSynchronize();

	// Decompression
	Vectorfield<float> vf_rec{};
	t0 = std::chrono::high_resolution_clock::now();
	auto tgpu_decompression = decompressWithCudaCompress(vf_comp, vf_rec);
	t1 = std::chrono::high_resolution_clock::now();
	auto tcpu_decompression = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	CompressionEfficiency(vf_comp.hData, vf_src.NumVectors(4), vf_src.ChannelCount);
	printf("\n");

	// Integration
	IntegrationConfig integrationConf{};
	integrationConf.Seeds = uint3{ 25, 25, 12 };
	integrationConf.CalculateStride(Dimensions);
	integrationConf.Steps = 20100;
	integrationConf.dt = 0.01f;

	dim3 threadsPerBlock(2, 2, 2);
	dim3 numBlocks((integrationConf.Seeds.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
		(integrationConf.Seeds.y + threadsPerBlock.y - 1) / threadsPerBlock.y,
		(integrationConf.Seeds.z + threadsPerBlock.z - 1) / threadsPerBlock.z);

	printf("Launching Particle Tracer\n");
	printf("Data Dimensinos:    (%i, %i, %i, %i)\n", Dimensions.x, Dimensions.y, Dimensions.z, Dimensions.w);
	printf("Seeds:              (%i, %i, %i)\n", integrationConf.Seeds.x, integrationConf.Seeds.y, integrationConf.Seeds.z);
	printf("Steps:              %u\n", integrationConf.Steps);
	printf("Stepsize:           %.4f\n", integrationConf.dt);
	printf("Threads per Block:  (%i, %i, %i)\n", threadsPerBlock.x, threadsPerBlock.y, threadsPerBlock.z);
	printf("Number of Blocks:   (%i, %i, %i)\n", numBlocks.x, numBlocks.y, numBlocks.z);
	printf("...\n");

	t0 = std::chrono::high_resolution_clock::now();
	auto tgpu_traces = Launch3Texture1CIntegrationTest(vf_rec, integrationConf, numBlocks, threadsPerBlock);
	t1 = std::chrono::high_resolution_clock::now();
	auto tcpu_trace = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	double tgpu_trace = 0.0;
	for (const auto t : tgpu_traces)
	{
		tgpu_trace += t;
	}
	tgpu_trace /= tgpu_traces.size();

	printf("Time GPU Compression:      %.4fms\n", tgpu_compression);
	printf("Time GPU Decompression:    %.4fms\n", tgpu_decompression);
	printf("Time GPU Particle Tracing: %.4fms\n", tgpu_trace);
	printf("Time CPU Compression:      %ums\n", tcpu_compression);
	printf("Time CPU Decompression:    %ums\n", tcpu_decompression);
	printf("Time CPU Particle Tracing: %ums\n", tcpu_trace / tgpu_traces.size());

	// Debug
	printf("Results:\n");
	avg_error(vf_src.Data.data(), vf_rec.Data.data(), vf_src.NumVectors(), vf_src.ChannelCount, (vf_src.InterleavedXYZ ? 1 : vf_src.NumVectors(4)));
}



void CompressFile(const char* filepath, bool save_decomp, bool interleaved_vectors, int numDecompLvls, float quantSize, int compIters, int huffBits)
{
	PrintDeviceInformation();

	// Read dimensions of processed dataset
	uint4 Dimensions{ 0, 0, 0, 0 };

	// Check if dimensions are stored seperately
	std::string dims_path = std::filesystem::path(filepath).replace_extension("").string() + "_dims.raw";
	if (exists(std::filesystem::path(dims_path)))
	{
		std::fstream file;
		file.open(dims_path, std::ios::in | std::ios::binary);
		if (!file.good())
		{
			return;
		}
		file.read((char*)&Dimensions.x, sizeof(Dimensions));
		file.close();
	}

	std::fstream file;
	file.open(filepath, std::ios::in | std::ios::binary | std::ios::ate);
	if (!file.good())
	{
		return;
	}
	std::streamsize filesize = file.tellg();
	file.seekg(0, std::ios::beg);
	if (!exists(std::filesystem::path(dims_path)))
		file.read(reinterpret_cast<char*>(&Dimensions), sizeof(Dimensions));

	Vectorfield<float> vf_src{};
	vf_src.Dimensions = Dimensions;
	vf_src.ChannelCount = filesize / ((uint64_t)Dimensions.x * (uint64_t)Dimensions.y * (uint64_t)Dimensions.z * (uint64_t)Dimensions.w * sizeof(float));
	vf_src.InterleavedXYZ = false;

	CompVectorfield vf_comp{};
	vf_comp.hData.resize(vf_src.Dimensions.w);
	vf_comp.numDecompositionLevels = numDecompLvls;		// Too many decompositions may introduce artifacts
	vf_comp.quantizationStepSize = quantSize;	// Granularity of data precision reduction (impacts compression efficiency) def: 0.00136f
	vf_comp.compressIterations = compIters;			// May improve... something
	vf_comp.huffmanBits = huffBits;
	vf_comp.b_RLEOnlyOnLvl0 = true;

	printf("Decomposition Levels:   %i\n", vf_comp.numDecompositionLevels);
	printf("Quantization Step Size: %.6f\n", vf_comp.quantizationStepSize);
	printf("Compression Iterations: %u\n", vf_comp.compressIterations);
	printf("Huffman Bits:           %u\n", vf_comp.huffmanBits);
	printf("-----------------------------------------\n");
	printf("Data Dimensions:       (%i, %i, %i, %i)\n", Dimensions.x, Dimensions.y, Dimensions.z, Dimensions.w);
	printf("Data Size:             %.2f MB\n", vf_src.NumScalars() * sizeof(float) / 1000000.0);
	printf("Launching Compression\n");
	printf("...\n");

	// Debug: Read the dataset into CPU memory
	vf_src.Data.resize(vf_src.NumScalars());
	auto t0 = std::chrono::high_resolution_clock::now();
	file.read(reinterpret_cast<char*>(vf_src.Data.data()), filesize - sizeof(Dimensions));
	auto t1 = std::chrono::high_resolution_clock::now();
	auto tcpu_fileread = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	// Compression
	t0 = std::chrono::high_resolution_clock::now();
	auto tgpu_compression = compressWithCudaCompress<float>(filepath, vf_comp, interleaved_vectors);
	t1 = std::chrono::high_resolution_clock::now();
	auto tcpu_compression = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	// Wait for completion
	cudaDeviceSynchronize();

	if (true)
	{
		SaveCompressedVectorfield(vf_comp, filepath);
	}

	printf("Launching Decompression\n");
	printf("...\n");

	// Decompression
	Vectorfield<float> vf_rec{};
	t0 = std::chrono::high_resolution_clock::now();
	auto tgpu_decompression = decompressWithCudaCompress(vf_comp, vf_rec);
	t1 = std::chrono::high_resolution_clock::now();
	auto tcpu_decompression = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

	CompressionEfficiency(vf_comp.hData, vf_src.NumVectors(4), vf_src.ChannelCount);
	printf("\n");

	printf("Time (GPU) Compression:      %.4fms\n", tgpu_compression);
	printf("Time (GPU) Decompression:    %.4fms\n", tgpu_decompression);
	printf("Time (CPU) Compression:      %ums\n", tcpu_compression);
	printf("Time (CPU) Decompression:    %ums\n", tcpu_decompression);

	auto err = avg_error(vf_src.Data.data(), vf_rec.Data.data(), vf_src.NumVectors(), vf_src.ChannelCount, (vf_src.InterleavedXYZ ? 1 : vf_src.NumVectors(4)));

	const auto absolute_dataset_path = std::filesystem::absolute(filepath);

	if (save_decomp) {
		std::fstream file_out;
		const std::string save_path = fmt::format("{}/{}_cudaComp_{}_{}_{}_{}.raw",
			std::filesystem::path(filepath).parent_path().has_parent_path() ? std::filesystem::path(filepath).parent_path().string() : ".",
			absolute_dataset_path.filename().replace_extension("").string(),
			vf_comp.numDecompositionLevels,
			vf_comp.quantizationStepSize,
			vf_comp.compressIterations,
			vf_comp.huffmanBits
		);

		file_out.open(save_path, std::ios::out | std::ios::binary);
		if (!file_out.good())
		{
			printf("Saving decompression version failed.");
		}
		file_out.write(reinterpret_cast<char*>(&vf_rec.Dimensions), sizeof(vf_rec.Dimensions));

		if (interleaved_vectors)
		{
			std::vector<float> slice(vf_rec.Dimensions.x* vf_rec.Dimensions.y* vf_rec.Dimensions.z * vf_rec.ChannelCount);

			for (uint64_t c = 0; c < vf_rec.ChannelCount; c++)
			{
				for (uint64_t t = 0; t < vf_rec.Dimensions.w; t++)
				{
					for (uint64_t z = 0; z < vf_rec.Dimensions.z; z++)
					{
						for (uint64_t y = 0; y < vf_rec.Dimensions.y; y++)
						{
							for (uint64_t x = 0; x < vf_rec.Dimensions.x; x++)
							{
								size_t idx_slice = (z * (uint64_t)vf_rec.Dimensions.y * (uint64_t)vf_rec.Dimensions.x + y * (uint64_t)vf_rec.Dimensions.x + x) * vf_rec.ChannelCount + c;
								size_t idx_vf = ((uint64_t)vf_rec.Dimensions.w * (uint64_t)vf_rec.Dimensions.z * (uint64_t)vf_rec.Dimensions.y * (uint64_t)vf_rec.Dimensions.x) * c + (t * (uint64_t)vf_rec.Dimensions.z * (uint64_t)vf_rec.Dimensions.y * (uint64_t)vf_rec.Dimensions.x + z * (uint64_t)vf_rec.Dimensions.y * (uint64_t)vf_rec.Dimensions.x + y * (uint64_t)vf_rec.Dimensions.x + x);
							    //size_t idx_vf = (t * (uint64_t)vf_rec.Dimensions.z * (uint64_t)vf_rec.Dimensions.y * (uint64_t)vf_rec.Dimensions.x + z * (uint64_t)vf_rec.Dimensions.y * (uint64_t)vf_rec.Dimensions.x + y * (uint64_t)vf_rec.Dimensions.x + x) * (uint64_t)vf_rec.ChannelCount;

								slice[idx_slice] = vf_rec.Data[idx_vf];
							}


						}
					}

					file.write((char*)slice.data(), slice.size() * sizeof(float));
					std::fill(slice.begin(), slice.end(), 0);
				}
			}
		}
		else
		{
			file_out.write(reinterpret_cast<char*>(vf_rec.Data.data()), vf_rec.Data.size() * sizeof(float));
		}

		file_out.close();
	}
}
