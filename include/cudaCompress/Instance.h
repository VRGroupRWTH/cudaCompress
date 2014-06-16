#ifndef __TUM3D_CUDACOMPRESS__INSTANCE_H__
#define __TUM3D_CUDACOMPRESS__INSTANCE_H__


#include <cudaCompress/global.h>

#include <cuda_runtime.h>


namespace cudaCompress {

// A cudaCompress::Instance encapsulates all resources required for entropy coding.
// The resources are pre-allocated for efficiency.
class Instance;

// Create a new cudaCompress::Instance.
// cudaDevice is the index of the CUDA device to use; pass -1 to use the current (default) one.
// streamCountMax is the max number of symbol streams to be encoded at once.
// elemCountPerStreamMax is the max number of values per stream.
// codingBlockSize specifies the parallelization granularity for the Huffman coder.
//   Smaller values result in more parallelism, but may hurt the compression rate.
//   Default: 128
// log2HuffmanDistinctSymbolCountMax is the max number of bits in the input values for the Huffman coder.
//   Should preferably be <= 16, must be <= 24; large values will result in higher memory usage and reduced performance.
//   Default: 14
Instance* createInstance(
    int cudaDevice,
    uint streamCountMax,
    uint elemCountPerStreamMax,
    uint codingBlockSize = 0,
    uint log2HuffmanDistinctSymbolCountMax = 0);
// Destroy a cudaCompress::Instance created previously by createInstance.
void  destroyInstance(Instance* pInstance);

// Query Instance parameters.
int getInstanceCudaDevice(const Instance* pInstance);
uint getInstanceStreamCountMax(const Instance* pInstance);
uint getInstanceElemCountPerStreamMax(const Instance* pInstance);
uint getInstanceCodingBlockSize(const Instance* pInstance);
uint getInstanceLog2HuffmanDistinctSymbolCountMax(const Instance* pInstance);
bool getInstanceUseLongSymbols(const Instance* pInstance);

// Set a cudaStream to use for all cudaCompress kernels.
void setInstanceCudaStream(Instance* pInstance, cudaStream_t str);
cudaStream_t getInstanceCudaStream(const Instance* pInstance);

}


#endif
