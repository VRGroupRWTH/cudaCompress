#pragma once
#include "global.h"
#include "cudaUtil.h"

// Texture Helpers
cudaArray_t Create3DArray(int channelCount, uint4 Dimensions, cudaChannelFormatKind ChannelFormat = cudaChannelFormatKindFloat);
void UploadTo3DArray(const float* SrcPtr, cudaArray_t DstPtr);
cudaTextureObject_t Create3DArrayTexture(cudaArray_t ArrayPtr);
