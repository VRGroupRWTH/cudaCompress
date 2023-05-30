#pragma once
#include <vector_types.h>
#include <cuda_runtime.h>
#include <assert.h>

cudaArray_t Create3DArray(int channelCount, uint4 Dimensions, cudaChannelFormatKind ChannelFormat = cudaChannelFormatKindFloat)
{
	cudaChannelFormatDesc channelDesc;
	channelDesc.f = ChannelFormat;

	if (ChannelFormat == cudaChannelFormatKindFloat)
	{
		channelDesc.x = (channelCount > 0) * 32;
		channelDesc.y = (channelCount > 1) * 32;
		channelDesc.z = (channelCount > 2) * 32;
		channelDesc.w = (channelCount > 3) * 32;
	}

	if (ChannelFormat == cudaChannelFormatKindSignedBlockCompressed6H)
	{
		channelDesc.x = (channelCount > 0) * 16;
		channelDesc.y = (channelCount > 1) * 16;
		channelDesc.z = (channelCount > 2) * 16;
		channelDesc.w = (channelCount > 3) * 0;
	}

	if (ChannelFormat == cudaChannelFormatKindUnsignedBlockCompressed6H)
	{
		channelDesc.x = (channelCount > 0) * 16;
		channelDesc.y = (channelCount > 1) * 16;
		channelDesc.z = (channelCount > 2) * 16;
		channelDesc.w = (channelCount > 3) * 0;
	}

	cudaArray_t datasetArray;	// Opaque data buffer optimized for texture fetches
	auto size = make_cudaExtent(Dimensions.x, Dimensions.y, Dimensions.z);
	cudaSafeCall(cudaMalloc3DArray(&datasetArray, &channelDesc, size, cudaArrayDefault));

	return datasetArray;
}

void UploadTo3DArray(const float* SrcPtr, cudaArray_t DstPtr)
{
	cudaChannelFormatDesc channelDesc;
	cudaExtent extents;
	uint flags;
	cudaArrayGetInfo(&channelDesc, &extents, &flags, DstPtr);

	int channelCount = (channelDesc.x > 0) + (channelDesc.y > 0) + (channelDesc.z > 0) + (channelDesc.w > 0);
	uint3 Dimensions{ (uint)extents.width, (uint)extents.height, (uint)extents.depth };

	cudaMemcpy3DParms CopyParams{};
    auto Extents = make_cudaExtent(Dimensions.x, Dimensions.y, Dimensions.z);
	auto BytePerVector = channelCount * sizeof(float);
	
	CopyParams.srcPtr = make_cudaPitchedPtr((float*)SrcPtr, (size_t)Dimensions.x * BytePerVector, Dimensions.x, Dimensions.y);
	CopyParams.dstArray = DstPtr;
	CopyParams.extent = Extents;
	CopyParams.kind = cudaMemcpyHostToDevice;
	cudaSafeCall(cudaMemcpy3D(&CopyParams));
}

cudaTextureObject_t Create3DArrayTexture(cudaArray_t ArrayPtr)
{
	cudaChannelFormatDesc channelDesc;
	cudaExtent extents;
	uint flags;
	cudaArrayGetInfo(&channelDesc, &extents, &flags, ArrayPtr);
	int channelCount = (channelDesc.x > 0) + (channelDesc.y > 0) + (channelDesc.z > 0) + (channelDesc.w > 0);
	uint3 Dimensions{ (uint)extents.width, (uint)extents.height, (uint)extents.depth };

	// create texture object
	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.linear.devPtr = ArrayPtr;
	resDesc.res.linear.desc = channelDesc;
	resDesc.res.linear.sizeInBytes = (size_t)Dimensions.x * (size_t)Dimensions.y * (size_t)Dimensions.z * (size_t)(channelCount) * sizeof(float);
	//resDesc.res.linear.sizeInBytes = MemoryRequirements.size;

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.readMode = cudaReadModeElementType;
	texDesc.normalizedCoords = 0;
	texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;
	texDesc.addressMode[2] = cudaAddressModeClamp;
	texDesc.filterMode = cudaFilterModeLinear;
	
	cudaTextureObject_t tex = 0;
	cudaSafeCall(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

	return tex;
}

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

	void DeleteTexture(cudaTextureObject_t TextureObject)
	{
		assert(TextureObjects.size() == ArrayObjects.size() && "Unsymmetric number of texture objects and arrays!");

		int idx = 0;
		for (const auto texObjs : TextureObjects)
		{
			if (texObjs == TextureObject)
				break;

			idx++;
		}

		for (auto i = idx; i < TextureObjects.size() - 1; i++)
		{
			std::swap(TextureObjects[i], TextureObjects[i + 1]);
			std::swap(ArrayObjects[i], ArrayObjects[i + 1]);
		}

		cudaSafeCall(cudaDestroyTextureObject(TextureObjects.back()));
		cudaSafeCall(cudaFreeArray(ArrayObjects.back()));

	    TextureObjects.pop_back();
		ArrayObjects.pop_back();
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

	cudaTextureObject_t* GetDeviceTextureObjects(std::vector<uint> ids)
	{
		std::vector<cudaTextureObject_t> selected;
		for (const auto idx : ids)
		{
			assert(idx < TextureObjects.size() && "Textures out of bounds!");
			selected.push_back(TextureObjects[idx]);
		}

		cudaTextureObject_t* d_textures;
		CreateDevicePointer(&d_textures, ids.size() * sizeof(cudaTextureObject_t));
		cudaSafeCall(cudaMemcpy(d_textures, selected.data(), selected.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

	    DeviceTextureRefs.push_back(d_textures);

	    return d_textures;
	}

	cudaTextureObject_t* GetDeviceTextureObjects()
	{
		cudaTextureObject_t* d_textures;
		CreateDevicePointer(&d_textures, TextureObjects.size() * sizeof(cudaTextureObject_t));
		cudaSafeCall(cudaMemcpy(d_textures, TextureObjects.data(), TextureObjects.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

		DeviceTextureRefs.push_back(d_textures);

		return d_textures;
	}

	Textures3C GetDeviceTextureObjects3C()
	{
		assert((TextureObjects.size() % 3) == 0 && "Requested RGB as single Channel but did not match available textures");

		Textures3C d_textures{};

		auto NumTexturesPerChannel = TextureObjects.size() / 3;
		CreateDevicePointer(&d_textures.x, NumTexturesPerChannel * sizeof(cudaTextureObject_t));
		CreateDevicePointer(&d_textures.y, NumTexturesPerChannel * sizeof(cudaTextureObject_t));
		CreateDevicePointer(&d_textures.z, NumTexturesPerChannel * sizeof(cudaTextureObject_t));

		cudaMemcpy(d_textures.x, TextureObjects.data(), NumTexturesPerChannel * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_textures.y, TextureObjects.data() + 1 * NumTexturesPerChannel, NumTexturesPerChannel * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
		cudaMemcpy(d_textures.z, TextureObjects.data() + 2 * NumTexturesPerChannel, NumTexturesPerChannel * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);

		DeviceTextureRefs.push_back(d_textures.x);
		DeviceTextureRefs.push_back(d_textures.y);
		DeviceTextureRefs.push_back(d_textures.z);

		return d_textures;
	}

	cudaArray_t GetArrayObject(uint idx)
	{
		assert(idx < ArrayObjects.size() && "Arrays out of bounds!");
		return ArrayObjects[idx];
	}

	~TextureManager()
	{
	    for (auto& tex : TextureObjects)
	    {
			cudaSafeCall(cudaDestroyTextureObject(tex));
	    }

		for (auto& array : ArrayObjects)
		{
			cudaSafeCall(cudaFreeArray(array));
		}

		for (auto& texref : DeviceTextureRefs)
		{
			cudaSafeCall(cudaFree(texref));
		}
	}

private:
	uint64_t GetVRAM() const
	{
		int nDevices;
		cudaGetDeviceCount(&nDevices);

		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, 0);

		return props.totalGlobalMem;
	}

public:
	std::vector<cudaTextureObject_t> TextureObjects;
	std::vector<cudaArray_t> ArrayObjects;

private:
	std::vector<cudaTextureObject_t*> DeviceTextureRefs;
};
