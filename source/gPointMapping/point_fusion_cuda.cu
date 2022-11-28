//-----------------------------------------------------------------------------
// NVIDIA(R) GVDB VOXELS
// Copyright 2018 NVIDIA Corporation
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#define CUDA_KERNEL
#include "point_fusion_cuda.cuh"
#include "cutil_math.h"			// cutil32.lib
#include <string.h>
#include <assert.h>

struct ALIGN(16) Obj {
	float3		pos;
	float3		size;
	float3		loc;
	uint		clr;
};

struct ALIGN(16) ScanInfo {
	int*		objGrid;
	int*		objCnts;
	Obj*		objList;
	float*		pxlList;
	float3*		pntList;
	uint*		pntClrs;
	int3		gridRes;
	float3		gridSize;	
	float3		cams;
	float3		camu;
	float3		camv;
	float3		camn;
	float		maxDist;
	uint*		rnd_seeds;
};
__device__ ScanInfo		scan;
__device__ int			pntout;

// Generate random unsigned int in [0, 2^24)
static __host__ __device__ __inline__ unsigned int lcg(unsigned int &prev)
{
  const unsigned int LCG_A = 1664525u;
  const unsigned int LCG_C = 1013904223u;
  prev = (LCG_A * prev + LCG_C);
  return prev & 0x00FFFFFF;
}
static __host__ __device__ __inline__ float rnd(unsigned int &prev)
{
  return ((float) lcg(prev) / (float) 0x01000000);
}

// Get view ray
inline __device__ float3 getViewRay ( float x, float y )
{
	float3 v = x*scan.camu + y*scan.camv + scan.cams;  
	return normalize(v);
}

#define NOHIT			1.0e10f

// Ray box intersection
inline __device__ float3 rayBoxIntersect ( float3 rpos, float3 rdir, float3 vmin, float3 vmax )
{
	register float ht[8];
	ht[0] = (vmin.x - rpos.x)/rdir.x;
	ht[1] = (vmax.x - rpos.x)/rdir.x;
	ht[2] = (vmin.y - rpos.y)/rdir.y;
	ht[3] = (vmax.y - rpos.y)/rdir.y;
	ht[4] = (vmin.z - rpos.z)/rdir.z;
	ht[5] = (vmax.z - rpos.z)/rdir.z;
	ht[6] = fmax(fmax(fmin(ht[0], ht[1]), fmin(ht[2], ht[3])), fmin(ht[4], ht[5]));
	ht[7] = fmin(fmin(fmax(ht[0], ht[1]), fmax(ht[2], ht[3])), fmax(ht[4], ht[5]));	
	ht[6] = (ht[6] < 0 ) ? 0.0 : ht[6];
	return make_float3( ht[6], ht[7], (ht[7]<ht[6] || ht[7]<0) ? NOHIT : 0 );
}

#define COLOR(r,g,b)	( (uint((b)*255.0f)<<16) | (uint((g)*255.0f)<<8) | uint((r)*255.0f) ) 

float3 __device__ __inline__ jitter_sample ()
{	 
	uint index = (threadIdx.y % 128) * 128 + (threadIdx.x % 128);
    unsigned int seed  = scan.rnd_seeds[ index ]; 
    float uu = rnd( seed );
    float vv = rnd( seed );
	float ww = rnd( seed );   
	scan.rnd_seeds[ index ] = seed;
    return make_float3(uu,vv,ww);
}

extern "C" __global__ void scanBuildings ( float3 pos, int3 res, int num_obj, float tmax )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;

	//float3 jit = jitter_sample();
	float3 dir = getViewRay( float(x)/float(res.x), float(y)/float(res.y) );
	
	int gcell = int(pos.z/scan.gridSize.y) * scan.gridRes.x + int(pos.x/scan.gridSize.x);
	if ( gcell < 0 || gcell > scan.gridRes.x*scan.gridRes.y)  return;

	Obj* bldg;
	float3 t, tnearest;
	uint clr = 0;

	tnearest.x = NOHIT;	

	//for (int n=0; n < scan.objCnts[gcell]; n++) {
//		bldg = scan.objList + (scan.objGrid[gcell] + n);

	for (int n=0; n < num_obj; n++) {
		bldg = scan.objList + n;
		if ( bldg != 0 ) {
			t = rayBoxIntersect ( pos, dir, bldg->pos, bldg->pos + bldg->size );
			if ( t.x < tnearest.x && t.x < tmax && t.z != NOHIT ) {
				tnearest = t;
				clr = bldg->clr;
			}
		}
	}
	if ( tnearest.x == NOHIT || tnearest.x > scan.maxDist) { 
		scan.pxlList[ y*res.x + x] = 0.0; 
		scan.pntClrs[ y*res.x + x] = 0;	
		return; 
	}

	atomicAdd(&pntout, 1);
	
	float3 hitPos = pos + tnearest.x * dir;
	float n = dot(hitPos - pos, scan.camn) / dot(scan.camn, scan.camn); // TODO:Precalc second dot product
	scan.pxlList[ y*res.x + x] = n;
	scan.pntClrs[ y*res.x + x] = clr;	
}


extern "C" __global__ void convertToPC (int3 res, float4 row1, float4 row2, float4 row3, float4 row4) {	
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;

	float dist = -scan.pxlList[ y*res.x + x];

	if (dist == 0) {
		scan.pntList[ y*res.x + x] = make_float3(0, 0, 0);
		return;
	}

	float4 pos2d = make_float4((float)(res.x - x)*dist, (float)y*dist, dist, 1.0f);
	float4 pos3d;
	pos3d.x = dot(row1, pos2d);
	pos3d.y = dot(row2, pos2d);
	pos3d.z = dot(row3, pos2d);
	pos3d.w = dot(row4, pos2d);
	
	scan.pntList[ y*res.x + x] = make_float3(pos3d.x/pos3d.w, pos3d.y/pos3d.w, pos3d.z/pos3d.w);

}

/*__device__ float3 cameraPos; 	// camera origin
__device__ float3 cameraX; 		// pointing 1m right of the camera
__device__ float3 cameraY; 		// pointing 1m above the camera
__device__ float3 cameraZ; 		// pointing 1m in front of the camera
__device__ float* distances;*/





