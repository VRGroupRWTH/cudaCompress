#pragma once
#define PI 3.14159265359
#include <glm/vec3.hpp>
#include <glm/gtx/compatibility.hpp>
#include "cudaUtil.h"


// The test dataset consists of easily debuggable numbers which can be changed here in source code if necessary.
// The parameter dynamic determines whether the vector components value is static or if it changes with the position in the vectorfield.
template <typename T>
void GenerateTestDataset(T* dst, size_t DimX, size_t DimY, size_t DimZ, size_t DimT, int channels, bool interleaved, bool dynamic = true)
{
	uint64_t offset = 0;
	T ValX = 0.1;
	T ValY = 50.2;
	T ValZ = 100.3;
	T ValW = 150.4;
	bool AddPosition = true;

	for (auto t = 0; t < DimT; t++)
	{
		for (auto z = 0; z < DimZ; z++)
		{
			for (auto y = 0; y < DimY; y++)
			{
				for (auto x = 0; x < DimX; x++)
				{
					if (interleaved)
					{
						if (channels > 0)
							dst[offset + 0] = ValX + x * dynamic;
						if (channels > 1)
							dst[offset + 1] = ValY + y * dynamic;
						if (channels > 2)
							dst[offset + 2] = ValZ + z * dynamic;
						if (channels > 3)
							dst[offset + 3] = ValW + t * dynamic;

						offset += channels;
					}
					else
					{
						if (channels > 0)
							dst[offset + 0 * DimX * DimY * DimZ * DimT] = ValX + x * dynamic;
						if (channels > 1)
							dst[offset + 1 * DimX * DimY * DimZ * DimT] = ValY + y * dynamic;
						if (channels > 2)
							dst[offset + 2 * DimX * DimY * DimZ * DimT] = ValZ + z * dynamic;
						if (channels > 3)
							dst[offset + 3 * DimX * DimY * DimZ * DimT] = ValW + t * dynamic;

						offset += 1;
					}
				}
			}
		}
	}
}

template <typename T>
std::vector<T> GenerateTestDataset(size_t DimX, size_t DimY, size_t DimZ, size_t DimT, int channels, bool interleaved)
{
	std::vector<T> dataset(DimX * DimY * DimZ * DimT * channels);

	GenerateTestDataset(dataset.data(), DimX, DimY, DimZ, DimT, channels, interleaved);

	return dataset;
}

template <typename T>
std::vector<T> GenerateTestDataset(uint4 dims, int channels, bool interleaved)
{
	return GenerateTestDataset<T>(dims.x, dims.y, dims.z, dims.w, channels, interleaved);
}

// Generates a dataset after the ABC flow as stated in https://doi.org/10.1109/MCG.2008.106
// This dataset is best generated as float or double.
// The dimension parameters are offsetted to show a more interesting ROI.
// The parameter interleaved determines whether (true) the vectors are saved as xyzw,xyzw,...
// or (false) xx...,yy...,zz...,ww...
template <typename T>
void GenerateABCDataset(T* dst, int DimX, int DimY, int DimZ, int DimT, int channels, bool interleaved)
{
	// ROI selection
	int DimX0 = -100;
	int DimX1 = DimX0 + DimX;
	int DimY0 = 0;
	int DimY1 = DimY0 + DimY;
	int DimZ0 = -100;
	int DimZ1 = DimZ0 + DimZ;
	int DimT0 = 0;
	int DimT1 = DimT0 + DimT;

	// Initializing good ABC parameters which distributes the flow field dynamics over a larger field
	const float A_PARAM = glm::sqrt(3);
	const float B_PARAM = glm::sqrt(2);
	const float C_PARAM = 1.0;
	const float c_pos = 0.05;
	const float c_t1 = 0.05;
	const float c_t2 = 0.01;

	uint64_t offset = 0;
	for (auto t = DimT0; t < DimT1; t++) {
		const float time = t;
		const float a_coeff = A_PARAM + c_t1 * time * glm::sin(PI * time * c_t2);

		for (auto z = DimZ0; z < DimZ1; z++) {
			for (auto y = DimY0; y < DimY1; y++) {
				for (auto x = DimX0; x < DimX1; x++) {
					// (discrete) position in the dataset
					const glm::vec3 pos(x, y, z);
					// velocity at this position
					const glm::vec3 velo(
						a_coeff * glm::sin(pos.z * c_pos) + B_PARAM * glm::cos(pos.y * c_pos),
						B_PARAM * glm::sin(pos.x * c_pos) + C_PARAM * glm::cos(pos.z * c_pos),
						C_PARAM * glm::sin(pos.y * c_pos) + a_coeff * glm::cos(pos.x * c_pos)
					);

					// save in data buffer
					if (interleaved)
					{
						if (channels > 0)
						    dst[offset + 0] = velo.x;
						if (channels > 1)
						    dst[offset + 1] = velo.y;
						if (channels > 2)
						    dst[offset + 2] = velo.z;
						if (channels > 3)
							dst[offset + 3] = 1.0f;

						offset += channels;
					}
					else
					{
						if (channels > 0)
						    dst[offset + 0 * DimX * DimY * DimZ * DimT] = velo.x;
						if (channels > 1)
						    dst[offset + 1 * DimX * DimY * DimZ * DimT] = velo.y;
						if (channels > 2)
						    dst[offset + 2 * DimX * DimY * DimZ * DimT] = velo.z;
						if (channels > 3)
							dst[offset + 3 * DimX * DimY * DimZ * DimT] = 1.0f;

						offset += 1;
					}
				}
			}
		}
	}
}

template <typename T>
std::vector<T> GenerateABCDataset(size_t DimX, size_t DimY, size_t DimZ, size_t DimT, int channels, bool interleaved)
{
	std::vector<T> dataset(DimX * DimY * DimZ * DimT * (size_t)channels);

	GenerateABCDataset(dataset.data(), DimX, DimY, DimZ, DimT, channels, interleaved);

	return dataset;
}

template <typename T>
std::vector<T> GenerateABCDataset(uint4 dims, int channels, bool interleaved)
{
	return GenerateABCDataset<T>(dims.x, dims.y, dims.z, dims.w, channels, interleaved);
}