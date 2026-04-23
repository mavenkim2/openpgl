#pragma once

#include <cuda.h>

// float2
__device__ __host__ inline float2 make_float2(const float a)
{
    return make_float2(a, a);
}

// float3
__device__ __host__ inline float3 make_float3(const int3 &a)
{
    return make_float3((float)a.x, (float)a.y, (float)a.z);
}

__device__ __host__ inline float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ __host__ inline float3 operator+(const float3 &a, float b)
{
    return make_float3(a.x + b, a.y + b, a.z + b);
}

__device__ __host__ inline float3 &operator+=(float3 &a, const float3 &b)
{
    a = a + b;
    return a;
}

__device__ __host__ inline float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ __host__ inline float3 operator-(const float3 &a, float b)
{
    return make_float3(a.x - b, a.y - b, a.z - b);
}

__device__ __host__ inline float3 operator*(const float3 &a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ __host__ inline float3 operator*(const float3 &a, const float3 &b)
{
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ __host__ inline float3 operator*(float a, const float3 &b)
{
    return b * a;
}

__device__ __host__ inline float3 operator/(const float3 &a, const float3 &b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ __host__ inline float3 operator/(const float3 &a, float b)
{
    return make_float3(a.x / b, a.y / b, a.z / b);
}

__device__ __host__ inline float3 operator/=(float3 &a, float3 &b)
{
    a = a / b;
    return a;
}

__device__ __host__ inline float Dot(const float3 &a, const float3 &b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ __host__ inline float3 Floor(const float3 &value)
{
    return make_float3(floorf(value.x), floorf(value.y), floorf(value.z));
}

__device__ __host__ inline float3 Ceil(const float3 &value)
{
    return make_float3(ceilf(value.x), ceilf(value.y), ceilf(value.z));
}

__device__ __host__ inline float3 Max(const float3 &value, float floorValue)
{
    return make_float3(
        fmaxf(value.x, floorValue), fmaxf(value.y, floorValue), fmaxf(value.z, floorValue));
}

__device__ __host__ inline float3 Ldexp(const float3 &value, int exponent)
{
    return make_float3(
        ldexpf(value.x, exponent), ldexpf(value.y, exponent), ldexpf(value.z, exponent));
}

__device__ __host__ inline float3 Ldexp(const float3 &value, const int3 &exponent)
{
    return make_float3(
        ldexpf(value.x, exponent.x), ldexpf(value.y, exponent.y), ldexpf(value.z, exponent.z));
}

// int3
__device__ __host__ inline int3 make_int3(const float3 &value)
{
    return make_int3(int(value.x), int(value.y), int(value.z));
}

// uint3
__device__ __host__ inline uint3 make_uint3(const float3 &value)
{
    return make_uint3(uint32_t(value.x), uint32_t(value.y), uint32_t(value.z));
}

__device__ __host__ inline uint3 operator+(const uint3 &a, uint32_t b)
{
    return make_uint3(a.x + b, a.y + b, a.z + b);
}