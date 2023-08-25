#pragma once
#include <assert.h>
#include <vector>
#include "global.h"
#include "cudaUtil.h"

template <typename T>
void CreateDevicePointer(T** d_ptr, size_t bytes)
{
	cudaSafeCall(cudaMalloc(d_ptr, bytes));
	cudaSafeCall(cudaMemset(*d_ptr, 0, bytes));
}

struct DatasetInterface
{
	uint4 Dimensions = uint4{ 0, 0, 0, 0 };
	int ChannelCount = 0;
	bool InterleavedXYZ = true;
	size_t Bytes = 0;
	bool Device = false;

	size_t NumVectors(int dimension = 4) const
	{
		if (dimension == 4)
			return (size_t)Dimensions.x * (size_t)Dimensions.y * (size_t)Dimensions.z * (size_t)Dimensions.w;

		if (dimension == 3)
			return (size_t)Dimensions.x * (size_t)Dimensions.y * (size_t)Dimensions.z;

		if (dimension == 2)
			return (size_t)Dimensions.x * (size_t)Dimensions.y;

		if (dimension == 1)
			return (size_t)Dimensions.x;

		return 0;
	}

	size_t NumScalars(int dimension = 4) const
	{
		return NumVectors(dimension) * (size_t)ChannelCount;
	}
};


template <typename T>
struct Vectorfield : public DatasetInterface
{
	std::vector<T> Data;
};

struct CompVectorfield : public DatasetInterface
{
	// time<channel<data>>
	std::vector<std::vector<std::vector<uint>>> hData;
	std::vector<std::vector<uint*>> dData;

	int compressIterations = 10;
	float quantizationStepSize = 0.00136f;
	int numDecompositionLevels = 2;
	int huffmanBits = 0;
	bool b_RLEOnlyOnLvl0 = true;

	size_t DecompressedBytes()
	{
		return NumScalars() * sizeof(float);
	}
};

struct IntegrationConfig
{
	uint3 Seeds;
	float3 SeedsStride;
	uint Steps;
	float3 CellFactor;
	float dt;

	void CalculateStride(uint4 Dimensions)
	{
		SeedsStride = float3{ float(Dimensions.x) / Seeds.x, float(Dimensions.y) / Seeds.y, float(Dimensions.z) / Seeds.z };
	}

	size_t SeedsNum() const
	{
		return (size_t)Seeds.x * (size_t)Seeds.y * (size_t)Seeds.z;
	}
};

struct Textures3C
{
	cudaTextureObject_t* x;
	cudaTextureObject_t* y;
	cudaTextureObject_t* z;
};

struct Buffers3C
{
	float* x;
	float* y;
	float* z;
};


struct TextureManager
{
public:
	template <typename T>
	void RegisterTexture(const T* data_ptr, uint4 Dimensions, int ChannelCount)
	{
		ArrayObjects.push_back(Create3DArray(ChannelCount, Dimensions));
		UploadTo3DArray(data_ptr, ArrayObjects.back());
		TextureObjects.push_back(Create3DArrayTexture(ArrayObjects.back()));
	}

	size_t Num() const
	{
		assert(TextureObjects.size() == ArrayObjects.size() && "Unsymmetric number of texture objects and arrays!");
		return TextureObjects.size();
	}

	cudaTextureObject_t GetTextureObject(uint idx)
	{
		assert(idx < TextureObjects.size() && "Textures out of bounds!");
		return TextureObjects[idx];
	}

	cudaArray_t GetArrayObject(uint idx)
	{
		assert(idx < ArrayObjects.size() && "Arrays out of bounds!");
		return ArrayObjects[idx];
	}

	void DeleteTexture(cudaTextureObject_t TextureObject);
	cudaTextureObject_t* GetDeviceTextureObjects(std::vector<uint> ids);
	cudaTextureObject_t* GetDeviceTextureObjects();
	Textures3C GetDeviceTextureObjects3C();
	
	~TextureManager();

private:
	uint64_t GetVRAM() const;

public:
	std::vector<cudaTextureObject_t> TextureObjects;
	std::vector<cudaArray_t> ArrayObjects;

private:
	std::vector<cudaTextureObject_t*> DeviceTextureRefs;
};


// Integration helpers (CPU equivalents of GPU implementations)
inline float4 lerp_CPU(float4 a, float4 b, float t);
float4 buf3D_CPU(float4* data_ptr, uint4 Dimensions, float x, float y, float z, uint t);
float3 buf4D_CPU(float4* data_ptr, uint4 Dimensions, float x, float y, float z, float t);
float3 rk4_CPU(float4* data, uint4 Dimensions, float3 pos, float dt, float time);
bool checkParticleValid_CPU(float3 particlePos, uint4 fieldDims, float time);
void traceParticles_CPU(float4* data, uint4 Dimensions, IntegrationConfig IntegrationConf, uint3 particleId3D);
