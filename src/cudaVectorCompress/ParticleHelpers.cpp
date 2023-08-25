#include "ParticleHelpers.h"
#include <cstring>
#include "helper_math.h"

inline float4 lerp_CPU(float4 a, float4 b, float t)
{
	return a + t * (b - a);
}

float4 buf3D_CPU(float4* data_ptr, uint4 Dimensions, float x, float y, float z, uint t)
{
	// P0 & P1 are reference points which span a cube that contains the given position:
	//   P0: Strictly floored position, always valid
	//   P1: Strictly ceiled position, may exceed dataset boundaries
	int3 P0 = int3{ max((int)x, 0),
					max((int)y, 0),
					max((int)z, 0) };
	int3 P1 = int3{ min(P0.x + 1, (int)Dimensions.x - 1),
					min(P0.y + 1, (int)Dimensions.y - 1),
					min(P0.z + 1, (int)Dimensions.z - 1) };
	t = min(t, Dimensions.w - 1);

	// To prevent indexing outside of the buffer
	uint maxX = Dimensions.x - 1;
	uint maxY = Dimensions.y - 1;
	uint maxZ = Dimensions.z - 1;

	// Interpolation factor within [0.0, 1.0] in all directions
	float lam_x = fracf(x);
	float lam_y = fracf(y);
	float lam_z = fracf(z);

	uint elems2D = Dimensions.y * Dimensions.x;
	uint elems3D = elems2D * Dimensions.z;

	// Lower depth
	uint offsetT = min(t, Dimensions.w - 1) * elems3D;
	uint offset1 = offsetT + P0.z * elems2D + P0.y * Dimensions.x + P0.x;
	uint offset2 = offsetT + P0.z * elems2D + P0.y * Dimensions.x + P1.x;
	uint offset3 = offsetT + P0.z * elems2D + P1.y * Dimensions.x + P1.x;
	uint offset4 = offsetT + P0.z * elems2D + P1.y * Dimensions.x + P0.x;

	float4 c0 = data_ptr[offset1];
	float4 c1 = data_ptr[offset2];
	float4 c2 = data_ptr[offset3];
	float4 c3 = data_ptr[offset4];

	// Higher depth
	uint addZ = P1.z < (Dimensions.z - 1);
	offset1 += addZ * elems2D;
	offset2 += addZ * elems2D;
	offset3 += addZ * elems2D;
	offset4 += addZ * elems2D;

	float4 d0 = data_ptr[offset1];
	float4 d1 = data_ptr[offset2];
	float4 d2 = data_ptr[offset3];
	float4 d3 = data_ptr[offset4];

	// Bilinear interpolation (2D, near)
	float4 i0 = lerp(c0, c1, lam_x);
	float4 i1 = lerp(c3, c2, lam_x);
	float4 c = lerp(i0, i1, lam_y);

	// Bilinear interpolation (2D, far)
	float4 j0 = lerp(d0, d1, lam_x);
	float4 j1 = lerp(d3, d2, lam_x);
	float4 d = lerp(j0, j1, lam_y);

	// Trilinearly interpolated vector (3D)
	return lerp(c, d, lam_z);
}

float3 buf4D_CPU(float4* data_ptr, uint4 Dimensions, float x, float y, float z, float t)
{
	float4 vector0 = buf3D_CPU(data_ptr, Dimensions, x, y, z, uint(t));
	float4 vector1 = buf3D_CPU(data_ptr, Dimensions, x, y, z, uint(t) + 1);

	return make_float3(lerp_CPU(vector0, vector1, fracf(t)));
}

float3 rk4_CPU(float4* data, uint4 Dimensions, float3 pos, float dt, float time)
{
	float3 k1 = pos;
	float3 v1 = buf4D_CPU(data, Dimensions, k1.x, k1.y, k1.z, time);				// v1 = F(x)

	float3 k2 = pos + v1 * 0.5f * dt;
	float3 v2 = buf4D_CPU(data, Dimensions, k2.x, k2.y, k2.z, time + 0.5 * dt);		// v2 = F(x + 0.5 * v1 * dt)

	float3 k3 = pos + v2 * 0.5f * dt;
	float3 v3 = buf4D_CPU(data, Dimensions, k3.x, k3.y, k3.z, time + 0.5 * dt);		// v3 = F(x + 0.5 * v2 * dt)

	float3 k4 = pos + v3 * dt;
	float3 v4 = buf4D_CPU(data, Dimensions, k4.x, k4.y, k4.z, time + dt);			// v4 = F(x + 1.0 * v3 * dt)

	return float3{ v1.x + 2.0f * v2.x + 2.0f * v3.x + v4.x / 6.0f,
					v1.y + 2.0f * v2.y + 2.0f * v3.y + v4.y / 6.0f,
					v1.z + 2.0f * v2.z + 2.0f * v3.z + v4.z / 6.0f };
}

bool checkParticleValid_CPU(float3 particlePos, uint4 fieldDims, float time)
{
	if (particlePos.x > (fieldDims.x - 1) || particlePos.x < 0.0) {
		return false;
	}

	if (particlePos.y > (fieldDims.y - 1) || particlePos.y < 0.0) {
		return false;
	}

	if (particlePos.z > (fieldDims.z - 1) || particlePos.z < 0.0) {
		return false;
	}

	if (time > fieldDims.w) {
		return false;
	}

	return true;
}


void traceParticles_CPU(float4* data, uint4 Dimensions, IntegrationConfig IntegrationConf, uint3 particleId3D)
{
	// Simulate integration for a certain particle
	float3 posSum = make_float3(0.0f);
	float3 pos = make_float3(particleId3D) * IntegrationConf.SeedsStride;
	float3 velo = rk4_CPU(data, Dimensions, make_float3(0.0f), IntegrationConf.dt, 0.0f);

	float integrationTime = 0.0;

	int StepsRemaining = IntegrationConf.Steps;

	for (int step = 0; step < StepsRemaining; step++)
	{
		if (!checkParticleValid_CPU(pos, Dimensions, integrationTime))
		{
			break;
		}

		pos = pos + velo * IntegrationConf.dt;
		posSum += pos;
		velo = rk4_CPU(data, Dimensions, pos, IntegrationConf.dt, integrationTime);
		//velo = rk4_CPU(data, Dimensions, make_float3(0.0f), IntegrationConf.dt, integrationTime);
		integrationTime += IntegrationConf.dt;
	}

	printf("Particle 0 Sum (CPU): (%.2f, %.2f, %.2f)\n\n", posSum.x, posSum.y, posSum.z);
}

void TextureManager::DeleteTexture(cudaTextureObject_t TextureObject) {
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

cudaTextureObject_t* TextureManager::GetDeviceTextureObjects(std::vector<uint> ids)
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


cudaTextureObject_t* TextureManager::GetDeviceTextureObjects()
{
	cudaTextureObject_t* d_textures;
	CreateDevicePointer(&d_textures, TextureObjects.size() * sizeof(cudaTextureObject_t));
	cudaSafeCall(cudaMemcpy(d_textures, TextureObjects.data(), TextureObjects.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));

	DeviceTextureRefs.push_back(d_textures);

	return d_textures;
}


Textures3C TextureManager::GetDeviceTextureObjects3C()
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

TextureManager::~TextureManager()
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

uint64_t TextureManager::GetVRAM() const
{
	int nDevices;
	cudaGetDeviceCount(&nDevices);

	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, 0);

	return props.totalGlobalMem;
}