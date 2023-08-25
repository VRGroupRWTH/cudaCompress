#include "cudaHelpers.cuh"
#include <cuda_runtime.h>
#include <cstring>

cudaArray_t Create3DArray(int channelCount, uint4 Dimensions, cudaChannelFormatKind ChannelFormat)
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
