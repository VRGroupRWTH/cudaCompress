#include "cudaTests.cuh"

#include <vector>
#include <stdio.h>
#include <cstdint>
#include <chrono>
#include <cuda_runtime.h>

#include "CompressVolume.h"
#include "cudaUtil.h"
#include "ParticleHelpers.h"
#include "cudaTestKernel.cui"
#include "Datasets.h"

inline float4 lerp_CPU(float4 a, float4 b, float t)
{
	return a + t * (b - a);
}

float4 buf3D_CPU(float4* data_ptr, uint4 Dimensions, float x, float y, float z, uint t)
{
	// P0 & P1 are reference points which span a cube that contains the given position:
	//   P0: Strictly floored position, always valid
	//   P1: Strictly ceiled position, may exceed dataset boundaries
	int3 P0 = int3{ max((int)x, 0),
					max((int)y, 0),
					max((int)z, 0) };
	int3 P1 = int3{ min(P0.x + 1, (int)Dimensions.x - 1),
					min(P0.y + 1, (int)Dimensions.y - 1),
					min(P0.z + 1, (int)Dimensions.z - 1) };
	t = min(t, Dimensions.w - 1);

	// To prevent indexing outside of the buffer
	uint maxX = Dimensions.x - 1;
	uint maxY = Dimensions.y - 1;
	uint maxZ = Dimensions.z - 1;

	// Interpolation factor within [0.0, 1.0] in all directions
	float lam_x = fracf(x);
	float lam_y = fracf(y);
	float lam_z = fracf(z);

	uint elems2D = Dimensions.y * Dimensions.x;
	uint elems3D = elems2D * Dimensions.z;

	// Lower depth
	uint offsetT = min(t, Dimensions.w - 1) * elems3D;
	uint offset1 = offsetT + P0.z * elems2D + P0.y * Dimensions.x + P0.x;
	uint offset2 = offsetT + P0.z * elems2D + P0.y * Dimensions.x + P1.x;
	uint offset3 = offsetT + P0.z * elems2D + P1.y * Dimensions.x + P1.x;
	uint offset4 = offsetT + P0.z * elems2D + P1.y * Dimensions.x + P0.x;

	float4 c0 = data_ptr[offset1];
	float4 c1 = data_ptr[offset2];
	float4 c2 = data_ptr[offset3];
	float4 c3 = data_ptr[offset4];

	// Higher depth
	uint addZ = P1.z < (Dimensions.z - 1);
	offset1 += addZ * elems2D;
	offset2 += addZ * elems2D;
	offset3 += addZ * elems2D;
	offset4 += addZ * elems2D;

	float4 d0 = data_ptr[offset1];
	float4 d1 = data_ptr[offset2];
	float4 d2 = data_ptr[offset3];
	float4 d3 = data_ptr[offset4];

	// Bilinear interpolation (2D, near)
	float4 i0 = lerp(c0, c1, lam_x);
	float4 i1 = lerp(c3, c2, lam_x);
	float4 c = lerp(i0, i1, lam_y);

	// Bilinear interpolation (2D, far)
	float4 j0 = lerp(d0, d1, lam_x);
	float4 j1 = lerp(d3, d2, lam_x);
	float4 d = lerp(j0, j1, lam_y);

	// Trilinearly interpolated vector (3D)
	return lerp(c, d, lam_z);
}

float3 buf4D_CPU(float4* data_ptr, uint4 Dimensions, float x, float y, float z, float t)
{
	float4 vector0 = buf3D_CPU(data_ptr, Dimensions, x, y, z, uint(t));
	float4 vector1 = buf3D_CPU(data_ptr, Dimensions, x, y, z, uint(t) + 1);

	return make_float3(lerp_CPU(vector0, vector1, fracf(t)));
}

float3 rk4_CPU(float4* data, uint4 Dimensions, float3 pos, float dt, float time)
{
	float3 k1 = pos;
	float3 v1 = buf4D_CPU(data, Dimensions, k1.x, k1.y, k1.z, time);				// v1 = F(x)

	float3 k2 = pos + v1 * 0.5f * dt;
	float3 v2 = buf4D_CPU(data, Dimensions, k2.x, k2.y, k2.z, time + 0.5 * dt);		// v2 = F(x + 0.5 * v1 * dt)

	float3 k3 = pos + v2 * 0.5f * dt;
	float3 v3 = buf4D_CPU(data, Dimensions, k3.x, k3.y, k3.z, time + 0.5 * dt);		// v3 = F(x + 0.5 * v2 * dt)

	float3 k4 = pos + v3 * dt;
	float3 v4 = buf4D_CPU(data, Dimensions, k4.x, k4.y, k4.z, time + dt);			// v4 = F(x + 1.0 * v3 * dt)

	return float3{ v1.x + 2.0f * v2.x + 2.0f * v3.x + v4.x / 6.0f,
					v1.y + 2.0f * v2.y + 2.0f * v3.y + v4.y / 6.0f,
					v1.z + 2.0f * v2.z + 2.0f * v3.z + v4.z / 6.0f };
}

bool checkParticleValid_CPU(float3 particlePos, uint4 fieldDims, float time)
{
	if (particlePos.x > (fieldDims.x - 1) || particlePos.x < 0.0) {
		return false;
	}

	if (particlePos.y > (fieldDims.y - 1) || particlePos.y < 0.0) {
		return false;
	}

	if (particlePos.z > (fieldDims.z - 1) || particlePos.z < 0.0) {
		return false;
	}

	if (time > fieldDims.w) {
		return false;
	}

	return true;
}


void traceParticles_CPU(float4* data, uint4 Dimensions, IntegrationConfig IntegrationConf, uint3 particleId3D)
{
	// Simulate integration for a certain particle
	float3 posSum = make_float3(0.0f);
	float3 pos = make_float3(particleId3D) * IntegrationConf.SeedsStride;
	float3 velo = rk4_CPU(data, Dimensions, make_float3(0.0f), IntegrationConf.dt, 0.0f);

	float integrationTime = 0.0;

	int StepsRemaining = IntegrationConf.Steps;

	for (int step = 0; step < StepsRemaining; step++)
	{
		if (!checkParticleValid_CPU(pos, Dimensions, integrationTime))
		{
			break;
		}

		pos = pos + velo * IntegrationConf.dt;
		posSum += pos;
		velo = rk4_CPU(data, Dimensions, pos, IntegrationConf.dt, integrationTime);
		//velo = rk4_CPU(data, Dimensions, make_float3(0.0f), IntegrationConf.dt, integrationTime);
	    integrationTime += IntegrationConf.dt;
	}

	printf("Particle 0 Sum (CPU): (%.2f, %.2f, %.2f)\n\n", posSum.x, posSum.y, posSum.z);
}

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
float avg_error(T* data_ptr1, T* data_ptr2, size_t numVecs, int channelCount, size_t channelStride = 1)
{
	double error_x_avg = 0.0;
	double error_y_avg = 0.0;
	double error_z_avg = 0.0;

	for (auto i = 0; i < numVecs; i++)
	{
		if (channelCount > 0)
		    error_x_avg += abs(data_ptr1[i + 0 * channelStride] - data_ptr2[i + 0 * channelStride]);
		if (channelCount > 1)
		    error_y_avg += abs(data_ptr1[i + 1 * channelStride] - data_ptr2[i + 1 * channelStride]);
		if (channelCount > 2)
		    error_z_avg += abs(data_ptr1[i + 2 * channelStride] - data_ptr2[i + 2 * channelStride]);
	}

	error_x_avg /= numVecs;
	error_y_avg /= numVecs;
	error_z_avg /= numVecs;

	printf("Average Error: (%.4f, %.4f, %.4f) [(x, y, z)]\n", error_x_avg, error_y_avg, error_z_avg);
	return (error_x_avg + error_y_avg + error_z_avg) / 3.0;
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
	float3* d_OutPosBuffer = nullptr;
	float3* d_OutVeloBuffer = nullptr;
	uint32_t* d_TraceLengthBuffer = nullptr;

	CreateDevicePointer(&d_OutPosBuffer, maxNumPoints * sizeof(float3));
	CreateDevicePointer(&d_OutVeloBuffer, maxNumPoints * sizeof(float3));
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
	std::vector<float3> ParticlePositions(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<float3> ParticleVelocities(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<uint32_t> ParticleTraceLength(conf.SeedsNum());

	cudaSafeCall(cudaMemcpy(ParticlePositions.data(), d_OutPosBuffer, ParticlePositions.size() * sizeof(float3), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleVelocities.data(), d_OutVeloBuffer, ParticleVelocities.size() * sizeof(float3), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleTraceLength.data(), d_TraceLengthBuffer, ParticleTraceLength.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaDeviceSynchronize());

	// Read
	float3 ParticleSum = make_float3(0.0f);
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
double Launch3Texture1CIntegrationTest(const Vectorfield<T>& vf, const IntegrationConfig& conf, dim3 numBlocks, dim3 threadsPerBlock, int iterations = 2)
{
	TextureManager texManager{};

	auto elems3D = (size_t)vf.Dimensions.x * (size_t)vf.Dimensions.y * (size_t)vf.Dimensions.z;
	for (size_t tidx = 0; tidx < vf.Dimensions.w * vf.ChannelCount; tidx++)
	{
		texManager.RegisterTexture(vf.Data.data() + tidx * elems3D, vf.Dimensions, 1);
	}

	// Device Buffers
	float3* d_OutPosBuffer = nullptr;
	float3* d_OutVeloBuffer = nullptr;
	uint32_t* d_TraceLengthBuffer = nullptr;

	size_t maxNumPoints = conf.SeedsNum() * (conf.Steps + 1);
	CreateDevicePointer(&d_OutPosBuffer, maxNumPoints * sizeof(float3));
	CreateDevicePointer(&d_OutVeloBuffer, maxNumPoints * sizeof(float3));
	CreateDevicePointer(&d_TraceLengthBuffer, conf.SeedsNum() * sizeof(uint32_t));

	auto d_textures3C = texManager.GetDeviceTextureObjects3C();

	// Recording Events
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Kernel Execution
	cudaEventRecord(start);

    for (auto i = 0; i < iterations; i++)
	{
		seedParticles << <numBlocks, threadsPerBlock >> > (
			d_textures3C,
			vf.Dimensions, conf,
			d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);
		traceParticles << <numBlocks, threadsPerBlock >> > (
			d_textures3C,
			vf.Dimensions, conf,
			d_OutPosBuffer, d_OutVeloBuffer, d_TraceLengthBuffer);
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
	printf("3 Textures time: %.6f ms\n", time);

	// Device to Host copy
	std::vector<float3> ParticlePositions(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<float3> ParticleVelocities(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<uint32_t> ParticleTraceLength(conf.SeedsNum());

	cudaSafeCall(cudaMemcpy(ParticlePositions.data(), d_OutPosBuffer, ParticlePositions.size() * sizeof(float3), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleVelocities.data(), d_OutVeloBuffer, ParticleVelocities.size() * sizeof(float3), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleTraceLength.data(), d_TraceLengthBuffer, ParticleTraceLength.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaDeviceSynchronize());

	// Read
	float3 ParticleSum = make_float3(0.0f);
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
float Launch1BufferIntegrationTest(const Vectorfield<T>& vf, const IntegrationConfig& conf, dim3 numBlocks, dim3 threadsPerBlock, int iterations = 2)
{
	// Data Upload
	float4* d_dataset = nullptr;
	CreateDevicePointer(&d_dataset, vf.Data.size() * sizeof(float));
	cudaSafeCall(cudaMemcpy(d_dataset, vf.Data.data(), vf.Data.size() * sizeof(float), cudaMemcpyHostToDevice));

	// Device Buffers
	float3* d_OutPosBuffer = nullptr;
	float3* d_OutVeloBuffer = nullptr;
	uint32_t* d_TraceLengthBuffer = nullptr;

	size_t maxNumPoints = conf.SeedsNum() * (conf.Steps + 1);
	CreateDevicePointer(&d_OutPosBuffer, maxNumPoints * sizeof(float3));
	CreateDevicePointer(&d_OutVeloBuffer, maxNumPoints * sizeof(float3));
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
	std::vector<float3> ParticlePositions(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<float3> ParticleVelocities(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<uint32_t> ParticleTraceLength(conf.SeedsNum());

	cudaSafeCall(cudaMemcpy(ParticlePositions.data(), d_OutPosBuffer, ParticlePositions.size() * sizeof(float3), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleVelocities.data(), d_OutVeloBuffer, ParticleVelocities.size() * sizeof(float3), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleTraceLength.data(), d_TraceLengthBuffer, ParticleTraceLength.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaDeviceSynchronize());

	// Read
	float3 ParticleSum = make_float3(0.0f);
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
	float3* d_OutPosBuffer = nullptr;
	float3* d_OutVeloBuffer = nullptr;
	uint32_t* d_TraceLengthBuffer = nullptr;

	size_t maxNumPoints = conf.SeedsNum() * (conf.Steps + 1);
	CreateDevicePointer(&d_OutPosBuffer, maxNumPoints * sizeof(float3));
	CreateDevicePointer(&d_OutVeloBuffer, maxNumPoints * sizeof(float3));
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
	std::vector<float3> ParticlePositions(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<float3> ParticleVelocities(conf.SeedsNum() * (conf.Steps + 1));
	std::vector<uint32_t> ParticleTraceLength(conf.SeedsNum());

	cudaSafeCall(cudaMemcpy(ParticlePositions.data(), d_OutPosBuffer, ParticlePositions.size() * sizeof(float3), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleVelocities.data(), d_OutVeloBuffer, ParticleVelocities.size() * sizeof(float3), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaMemcpy(ParticleTraceLength.data(), d_TraceLengthBuffer, ParticleTraceLength.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
	cudaSafeCall(cudaDeviceSynchronize());

	// Read
	float3 ParticleSum = make_float3(0.0f);
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
	integrationConf.CellFactor = make_float3(1.0f);
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
void compressWithCudaCompress(const Vectorfield<T>& src_vf, CompVectorfield& dst_vf)
{
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
	    for (uint i = 0; i < dst_vf.compressIterations; i++)
	    {
		    for (size_t c = 0; c < src_vf.ChannelCount; c++)
		    {
			    compressVolumeFloat(GPUResources, CompVolumeResources, dp_Buffer_Images[c], src_vf.Dimensions.x, src_vf.Dimensions.y, src_vf.Dimensions.z, dst_vf.numDecompositionLevels, dst_vf.hData[t][c], dst_vf.quantizationStepSize, dst_vf.b_RLEOnlyOnLvl0);
		    }
	    }

		cudaSafeCall(cudaDeviceSynchronize());

		dst_vf.dData[t].resize(src_vf.ChannelCount);
		for (size_t c = 0; c < src_vf.ChannelCount; c++)
		{
			// If a gpu buffer already exists, destroy
			if (dst_vf.dData[t][c])
			{
				cudaSafeCall(cudaFree(dp_Buffer_Images[c]));
			}

			// create gpu buffer and copy content
			auto BytesCompressed = uint(dst_vf.hData[t][c].size()) * sizeof(uint);
		    CreateDevicePointer(&dst_vf.dData[t][c], BytesCompressed);
			cudaSafeCall(cudaMemcpy(dst_vf.dData[t][c], dst_vf.hData[t][c].data(), BytesCompressed, cudaMemcpyHostToDevice));
		}
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
}

template <typename T>
void decompressWithCudaCompress(CompVectorfield& src_vf, Vectorfield<T>& dst_vf)
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

    for (auto t = 0; t < src_vf.Dimensions.w; t++)
    {
	    // Reset memory to 0
	    for (size_t c = 0; c < src_vf.ChannelCount; c++)
	    {
		    cudaSafeCall(cudaMemset(dp_Buffer_Images[c], 0, src_vf.NumVectors(3) * sizeof(float)));
	    }


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
	    for (uint i = 0; i < src_vf.compressIterations; i++)
	    {
		    decompressVolumeFloatMultiChannel(GPUResources, CompVolumeResources, channels.data(), (uint)channels.size(), src_vf.Dimensions.x, src_vf.Dimensions.y, src_vf.Dimensions.z, src_vf.numDecompositionLevels, src_vf.b_RLEOnlyOnLvl0);
	    }

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
	avg_error(vf.Data.data(), r_vf.Data.data(), vf.NumVectors(4), (vf.InterleavedXYZ ? 1 : vf.NumVectors(3)));
}

void TestCompressedParticleKernel()
{
	PrintDeviceInformation();
	uint4 Dimensions{ 256 , 256, 256, 10 };

	IntegrationConfig integrationConf{};
	integrationConf.Seeds = uint3{ 10, 12, 12 };
	integrationConf.CalculateStride(Dimensions);
	integrationConf.Steps = 1000;
	integrationConf.CellFactor = make_float3(1.0f);
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
	avg_error(vf.Data.data(), r_vf.Data.data(), vf.NumVectors(4), (vf.InterleavedXYZ ? 1 : vf.NumVectors(3)));
}


