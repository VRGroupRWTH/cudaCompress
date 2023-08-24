#pragma once
#include <vector>
#include "ParticleHelpers.h"

void TestCUDAKernel();
void TestCudaCompressKernel();
void TestParticleKernel();
void TestTextureKernel();
void TestCompressedParticleKernel();

double Launch1Texture4CTest();
double Launch3Texture1CTest();
double Launch1BufferTest();
double Launch3BufferTest();

double Launch1Texture4CIntegrationTest();
std::vector<double> Launch3Texture1CIntegrationTest();
double Launch1BufferIntegrationTest();
double Launch3BufferIntegrationTest();

bool ReadCudaCompressedVF(CompVectorfield& vf_cmp, const char* filepath);
void TraceABC();
void TraceFile(const char* filepath, bool read_slicewise = false);
void CompressFile(const char* filepath, bool read_slicewise = true, bool save_decomp = true, int numDecompLvls = 2, float quantSize = 0.00136, int compIters = 10, int huffBits = 0);
void ReadFile(const char* filepath);
