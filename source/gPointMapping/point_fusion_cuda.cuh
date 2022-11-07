//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2018 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#ifndef DEF_KERN_CUDA
	#define DEF_KERN_CUDA

	#include <stdio.h>
	#include <math.h>

	#define CUDA_KERNEL
	typedef unsigned int		uint;
	typedef unsigned short int	ushort;
	typedef unsigned char		uchar;

	#define ALIGN(x)	__align__(x)
	
	
	extern "C" __global__ void scanBuildings ( float3 pos, int3 res, int num_obj, float tmax );
	extern "C" __global__ void convertToPC (int3 res, float4 row1, float4 row2, float4 row3, float4 row4);
	

#endif
