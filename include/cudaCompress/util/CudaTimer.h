#ifndef __TUM3D_CUDACOMPRESS__CUDA_TIMER_H__
#define __TUM3D_CUDACOMPRESS__CUDA_TIMER_H__


#include <cudaCompress/global.h>

#include <deque>
#include <map>
#include <string>
#include <vector>

#include <cuda_runtime.h>


namespace cudaCompress {

namespace util {

class CudaTimerResources
{
public:
    CudaTimerResources(bool enabled = false);
    ~CudaTimerResources();

    void setEnabled(bool enable = true) { m_isEnabled = enable; }

    const std::map<std::string, float>& getAccumulatedTimes(bool sync);
    void getAccumulatedTimes(std::vector<std::string>& names, std::vector<float>& times, bool sync);
    void reset();

    void record(const std::string& name);

private:
    struct NamedEvent
    {
        NamedEvent()
            : m_name(""), m_event(0) {}
        NamedEvent(const std::string& name, cudaEvent_t event)
            : m_name(name), m_event(event) {}
        void clear() { m_name.clear(); m_event = 0; }
        std::string m_name;
        cudaEvent_t m_event;
    };

    void collectEvents(bool sync);
    cudaEvent_t getEvent();

    bool                         m_isEnabled;

    std::deque<NamedEvent>       m_activeEvents;
    NamedEvent                   m_currentSection;

    std::vector<cudaEvent_t>     m_availableEvents;

    std::map<std::string, float> m_accumulatedTimes; //TODO order of events?
};


class CudaScopedTimer
{
public:
    CudaScopedTimer(CudaTimerResources& resources);
    ~CudaScopedTimer();

    // start timing the next section
    // call without params to close the current section
    void operator() (const std::string& name = std::string());

private:
    CudaTimerResources& m_resources;
    bool                m_sectionOpen;
};

}

}


#endif
