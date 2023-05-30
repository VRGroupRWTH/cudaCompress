#pragma once

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
double Launch3Texture1CIntegrationTest();
double Launch1BufferIntegrationTest();
double Launch3BufferIntegrationTest();