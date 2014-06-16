#ifndef __TUM3D_CUDACOMPRESS__TIMING_H__
#define __TUM3D_CUDACOMPRESS__TIMING_H__


#include <cudaCompress/global.h>

#include <string>
#include <vector>


namespace cudaCompress {

class Instance;

enum ETimingDetail
{
    TIMING_DETAIL_NONE,
    TIMING_DETAIL_LOW,
    TIMING_DETAIL_MEDIUM,
    TIMING_DETAIL_HIGH,
};

void setTimingDetail(Instance* pCudaCompressInstance, ETimingDetail detail);

void getTimings(Instance* pCudaCompressInstance, std::vector<std::string>& names, std::vector<float>& times);
void printTimings(Instance* pCudaCompressInstance);
void resetTimings(Instance* pCudaCompressInstance);

}


#endif
