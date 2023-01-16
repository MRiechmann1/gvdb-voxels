struct ALIGN(16) FrameInfo {
	float3*		pntList;
	uint*		pntClrs;
	int3		res;	
	float3		cams;
	float3		camu;
	float3		camv;
	float3 		pos;
	uint 		numPts;
	float  		maxDist; // in voxels number
	float  		minDist; // in voxels number
	float 		maxProb;
	float		minProb;
};
__device__ FrameInfo		frame;

// TODO: Optimize implementation -> reduce the number of branches (if and loops)


/*
 * Implementation of taken from point_fusion_cuda.cu rayBoxIntersection and if condition adapted
 * TODO: check why this is so much faster? No ifs? no register? -> no
 */
inline __device__ bool intersection (float3 box, float3 rpos, float3 rdir)
{
	register float ht[8];
	ht[0] = (box.x - rpos.x)/rdir.x;
	ht[1] = (box.x + 1.0f - rpos.x)/rdir.x;
	ht[2] = (box.y - rpos.y)/rdir.y;
	ht[3] = (box.y + 1.0f - rpos.y)/rdir.y;
	ht[4] = (box.z - rpos.z)/rdir.z;
	ht[5] = (box.z + 1.0f - rpos.z)/rdir.z;
	ht[6] = fmax(fmax(fmin(ht[0], ht[1]), fmin(ht[2], ht[3])), fmin(ht[4], ht[5]));
	ht[7] = fmin(fmin(fmax(ht[0], ht[1]), fmax(ht[2], ht[3])), fmax(ht[4], ht[5]));	
	ht[6] = (ht[6] < 0 ) ? 0.0 : ht[6];
	return ht[7]>=ht[6] && ht[7]>=0 && ht[6]<=1.0f;
}

/*bool __device__ __inline__ intersection(float3 box, float3 origin,float3 ray) {
    double tmin = -INFINITY, tmax = INFINITY;

    if (ray.x != 0.0) {
        double tx1 = (box.x - origin.x)/ray.x;
        double tx2 = (box.x + 1 - origin.x)/ray.x;

        tmin = max(tmin, min(tx1, tx2));
        tmax = min(tmax, max(tx1, tx2));
    }

    if (ray.y != 0.0) {
        double ty1 = (box.y - origin.y)/ray.y;
        double ty2 = (box.y + 1 - origin.y)/ray.y;

        tmin = max(tmin, min(ty1, ty2));
        tmax = min(tmax, max(ty1, ty2));
    }

	if (ray.z != 0.0) {
        double tz1 = (box.z - origin.z)/ray.z;
        double tz2 = (box.z + 1 - origin.z)/ray.z;

        tmin = max(tmin, min(tz1, tz2));
        tmax = min(tmax, max(tz1, tz2));
    }

    return tmax >= tmin && tmax <= 1.0;
}*/

bool __device__ __inline__ inFrustrum(float3 koef) {
	return koef.x <= 1.0 && koef.x >= 0 && 
		koef.y <= koef.x && koef.y >= 0 &
		koef.z <= koef.x & koef.z >= 0;
}

float3 __device__ __inline__ minValue(float3 *koef) {
	float3 minV;
	/*minV.x = min(koef[0].x, koef[1].x);
	minV.x = min(minV.x, koef[2].x);
	minV.x = min(minV.x, koef[3].x);
	minV.x = min(minV.x, koef[4].x);
	minV.x = min(minV.x, koef[5].x);
	minV.x = min(minV.x, koef[6].x);
	minV.x = min(minV.x, koef[7].x);
	minV.x = max(minV.x, 0.0);*/

	minV.y = max(min(min(min(koef[0].y / koef[0].x, koef[1].y / koef[1].x), min(koef[2].y / koef[2].x, koef[3].y / koef[3].x)), 
				 min(min(koef[4].y / koef[4].x, koef[5].y / koef[5].x), min(koef[6].y / koef[6].x, koef[7].y / koef[7].x))), 0.0f);

	minV.z = max(min(min(min(koef[0].z / koef[0].x, koef[1].z / koef[1].x), min(koef[2].z / koef[2].x, koef[3].z / koef[3].x)), 
				 min(min(koef[4].z / koef[4].x, koef[5].z / koef[5].x), min(koef[6].z / koef[6].x, koef[7].z / koef[7].x))), 0.0f);

	return minV;
}

float3 __device__ __inline__ maxValue(float3 *koef) {
	float3 minV;
	/*minV.x = max(koef[0].x / koef[0].x, koef[1].x / koef[0].x);
	minV.x = max(minV.x, koef[2].x) / koef[2].x;
	minV.x = max(minV.x, koef[3].x / koef[3].x);
	minV.x = max(minV.x, koef[4].x / koef[4].x);
	minV.x = max(minV.x, koef[5].x / koef[5].x);
	minV.x = max(minV.x, koef[6].x / koef[6].x);
	minV.x = max(minV.x, koef[7].x / koef[7].x);
	minV.x = min(minV.x, 1.0);*/

	minV.y = min(max(max(max(koef[0].y / koef[0].x, koef[1].y / koef[1].x), max(koef[2].y / koef[2].x, koef[3].y / koef[3].x)), 
				 	 max(max(koef[4].y / koef[4].x, koef[5].y / koef[5].x), max(koef[6].y / koef[6].x, koef[7].y / koef[7].x))), 1.0f);

	minV.z = min(max(max(max(koef[0].z / koef[0].x, koef[1].z / koef[1].x), max(koef[2].z / koef[2].x, koef[3].z / koef[3].x)), 
					 max(max(koef[4].z / koef[4].x, koef[5].z / koef[5].x), max(koef[6].z / koef[6].x, koef[7].z / koef[7].x))), 1.0f);

	return minV;
}

/*
 * Considers cams camu camv as rows 1-3 of the inverse volume equation fo the view frustrum
 */
extern "C" __global__ void gvdbUpdateMap ( VDBInfo* gvdb, int3 atlasRes, uchar chan,float p1, float p2, float p3  )
{
	float3 relPos, wpos, koef[8];
	GVDB_VOXUNPACKED

	// TODO: Filter voxel not in region of the camera compare voxel to the (max values of the the grid?)
	// TODO: change compute function to only walk over the voxels close to the camera

	if ( !getAtlasToWorld ( gvdb, atlasIdx, wpos )) return;
	wpos -= make_float3(0.5, 0.5, 0.5);

	/*
	 * Check if voxel is in range bounds
	 */
	relPos = wpos - frame.pos;
	koef[0].x = dot(frame.cams, relPos);
	koef[0].y = dot(frame.camu, relPos);
	koef[0].z = dot(frame.camv, relPos);

	relPos.x++;
	koef[1].x = dot(frame.cams, relPos);
	koef[1].y = dot(frame.camu, relPos);
	koef[1].z = dot(frame.camv, relPos);

	relPos.y++;;
	koef[2].x = dot(frame.cams, relPos);
	koef[2].y = dot(frame.camu, relPos);
	koef[2].z = dot(frame.camv, relPos);

	relPos.z++;
	koef[3].x = dot(frame.cams, relPos);
	koef[3].y = dot(frame.camu, relPos);
	koef[3].z = dot(frame.camv, relPos);

	relPos.y--;
	koef[4].x = dot(frame.cams, relPos);
	koef[4].y = dot(frame.camu, relPos);
	koef[4].z = dot(frame.camv, relPos);

	relPos.x--;
	koef[5].x = dot(frame.cams, relPos);
	koef[5].y = dot(frame.camu, relPos);
	koef[5].z = dot(frame.camv, relPos);

	relPos.y++;
	koef[6].x = dot(frame.cams, relPos);
	koef[6].y = dot(frame.camu, relPos);
	koef[6].z = dot(frame.camv, relPos);

	relPos.z--;
	koef[7].x = dot(frame.cams, relPos);
	koef[7].y = dot(frame.camu, relPos);
	koef[7].z = dot(frame.camv, relPos);

	relPos.y--;

	if (!inFrustrum(koef[0]) && !inFrustrum(koef[1]) && !inFrustrum(koef[2]) && !inFrustrum(koef[3]) &&
		!inFrustrum(koef[4]) && !inFrustrum(koef[5]) && !inFrustrum(koef[6]) && !inFrustrum(koef[7])) return;

	float3 minK = minValue(koef);
	float3 maxK = maxValue(koef);

	minK.y = floor(minK.y * (float) frame.res.x);
	minK.z = floor(minK.z * (float) frame.res.y);
	maxK.y = ceil(maxK.y * (float) frame.res.x);
	maxK.z = ceil(maxK.z * (float) frame.res.y);

	int hitCount = 0;
	int freeCount = 0;
	float4 avgClr = make_float4(0,0,0,0);
	for (int x = minK.y; x < maxK.y; x++) {
		for (int y = minK.z; y < maxK.z; y++) {
			int i = y*frame.res.x + x;
			// is a measurment inside?

			if (!(frame.pntList[i].x == 0 && frame.pntList[i].y == 0 && frame.pntList[i].z == 0)) {
				if (frame.pntList[i].x >= wpos.x && frame.pntList[i].x < wpos.x+1 &&
						frame.pntList[i].y >= wpos.y && frame.pntList[i].y < wpos.y+1 &&
						frame.pntList[i].z >= wpos.z && frame.pntList[i].z < wpos.z+1) {
					hitCount++;
					uchar4 clr = ((uchar4 *)frame.pntClrs)[i];
					avgClr += make_float4(clr.x, clr.y, clr.z, clr.w);
				} else if (hitCount == 0 && intersection(wpos, frame.pos, frame.pntList[i] - frame.pos)) {
					freeCount++;
				}
			} else {
				/* TODO: When no point could be measured the area from the measurment to maximum range is probably free*/
				// -> check if voxel intersects ray from camera to pixel at maximum distance
				freeCount++;
			}
		}
	}
	float v = surf3Dread<float>(gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	if (hitCount > 0) {
		uchar4 clr = surf3Dread<uchar4>(gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);
		avgClr += make_float4(clr.x, clr.y, clr.z, clr.w);
		hitCount++;
		clr = make_uchar4(avgClr.x / hitCount, avgClr.y / hitCount, avgClr.z / hitCount, avgClr.w / hitCount);
		surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);

		float prob = min(v + hitCount, frame.maxProb);
		surf3Dwrite( prob, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
		return;
	}
	if (freeCount > 0) {
		/*uchar4 clr = make_uchar4(125, 125, 125, 255);
		surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);*/

		// TODO:-> give a free/occupied prob update value to the function
		float prob = max(v - freeCount/100, frame.minProb);//6;//max(v - hitCount, frame.minProb); 
		surf3Dwrite( prob, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
		return;
	}
	return;

}


extern "C" __global__ void gvdbUpdateMapVoxRegion ( VDBInfo* gvdb, int numPnts, int3 atlasRes, int3 minRegion, int3 dimRegion )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= dimRegion.x || y >= dimRegion.y || z >= dimRegion.z) return;

	float3 wpos = make_float3(x + minRegion.x, y + minRegion.y, z + minRegion.z);
	float3 offs, vmin; uint64 nid;
	VDBNode* node = getNodeAtPoint ( gvdb, wpos, &offs, &vmin, &nid );				// find vdb node at point
	//printf("%d, %d, %d\n", minRegion.x, minRegion.y, minRegion.z);
	if (node == 0x0) return;
	float3 atlasIdx = offs + wpos - vmin;
	if (atlasIdx.x >= atlasRes.x || atlasIdx.y >= atlasRes.x || atlasIdx.z >= atlasRes.x) return;

	/*
	 * Check if voxel is in range bounds
	 */
	float3 relPos = wpos - frame.pos;
	float3 koef[8];
	koef[0].x = dot(frame.cams, relPos);
	koef[0].y = dot(frame.camu, relPos);
	koef[0].z = dot(frame.camv, relPos);

	relPos.x++;
	koef[1].x = dot(frame.cams, relPos);
	koef[1].y = dot(frame.camu, relPos);
	koef[1].z = dot(frame.camv, relPos);

	relPos.y++;;
	koef[2].x = dot(frame.cams, relPos);
	koef[2].y = dot(frame.camu, relPos);
	koef[2].z = dot(frame.camv, relPos);

	relPos.z++;
	koef[3].x = dot(frame.cams, relPos);
	koef[3].y = dot(frame.camu, relPos);
	koef[3].z = dot(frame.camv, relPos);

	relPos.y--;
	koef[4].x = dot(frame.cams, relPos);
	koef[4].y = dot(frame.camu, relPos);
	koef[4].z = dot(frame.camv, relPos);

	relPos.x--;
	koef[5].x = dot(frame.cams, relPos);
	koef[5].y = dot(frame.camu, relPos);
	koef[5].z = dot(frame.camv, relPos);

	relPos.y++;
	koef[6].x = dot(frame.cams, relPos);
	koef[6].y = dot(frame.camu, relPos);
	koef[6].z = dot(frame.camv, relPos);

	relPos.z--;
	koef[7].x = dot(frame.cams, relPos);
	koef[7].y = dot(frame.camu, relPos);
	koef[7].z = dot(frame.camv, relPos);

	relPos.y--;

	if (!inFrustrum(koef[0]) && !inFrustrum(koef[1]) && !inFrustrum(koef[2]) && !inFrustrum(koef[3]) &&
		!inFrustrum(koef[4]) && !inFrustrum(koef[5]) && !inFrustrum(koef[6]) && !inFrustrum(koef[7])) return;

	float3 minK = minValue(koef);
	float3 maxK = maxValue(koef);

	minK.y = floor(minK.y * (float) frame.res.x);
	minK.z = floor(minK.z * (float) frame.res.y);
	maxK.y = ceil(maxK.y * (float) frame.res.x);
	maxK.z = ceil(maxK.z * (float) frame.res.y);

	int hitCount = 0;
	int freeCount = 0;
	float4 avgClr = make_float4(0,0,0,0);
	for (int x = minK.y; x < maxK.y; x++) {
		for (int y = minK.z; y < maxK.z; y++) {
			int i = y*frame.res.x + x;
			// is a measurment inside?

			if (!(frame.pntList[i].x == 0 && frame.pntList[i].y == 0 && frame.pntList[i].z == 0)) {
				if (frame.pntList[i].x >= wpos.x && frame.pntList[i].x < wpos.x+1 &&
						frame.pntList[i].y >= wpos.y && frame.pntList[i].y < wpos.y+1 &&
						frame.pntList[i].z >= wpos.z && frame.pntList[i].z < wpos.z+1) {
					hitCount++;
					uchar4 clr = ((uchar4 *)frame.pntClrs)[i];
					avgClr += make_float4(clr.x, clr.y, clr.z, clr.w);
				} else if (hitCount == 0 && intersection(wpos, frame.pos, frame.pntList[i] - frame.pos)) {
					freeCount++;
				}
			} else {
				/* TODO: When no point could be measured the area from the measurment to maximum range is probably free*/
				// -> check if voxel intersects ray from camera to pixel at maximum distance
				freeCount++;
			}
		}
	}
	float v = surf3Dread<float>(gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	if (hitCount > 0) {
		uchar4 clr = surf3Dread<uchar4>(gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);
		avgClr += make_float4(clr.x, clr.y, clr.z, clr.w);
		hitCount++;
		clr = make_uchar4(avgClr.x / hitCount, avgClr.y / hitCount, avgClr.z / hitCount, avgClr.w / hitCount);
		surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);

		float prob = min(v + hitCount, frame.maxProb);
		surf3Dwrite( prob, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
		return;
	}
	if (freeCount > 0) {
		/*uchar4 clr = make_uchar4(125, 125, 125, 255);
		surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);*/

		// TODO:-> give a free/occupied prob update value to the function
		float prob = max(v - freeCount/100, frame.minProb);//6;//max(v - hitCount, frame.minProb); 
		surf3Dwrite( prob, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
		return;
	}
	return;
}

// Follow the implementation of insert point (identify voxels that need to be updated) -> check how the range is tested
// then perform the insetion

// Follow the implementation of scanBuilding (especially raxbox intersect), to implemnt ray casting based insertion
// Voxel based implementation see board

struct ALIGN(16) RaycastUpdate {
	float3*		pntList;
	uint*		pntClrs;
	int3		res;	
	float3		cams;
	float3		camu;
	float3		camv;
	float3 		pos;
	uint 		numPts;
	float*		voxelsCpy;
	int3		voxelCpyOffset;
	int3  		voxelCpyDim;
	float3*  	voxelCpyClr;
	float 		voxel_size;
};
__device__ RaycastUpdate		rayInfo;


/*
 * Considers cams camu camv the borders vectors of the view frustrum
 */
extern "C" __global__ void gvdbUpdateMapPC ( int3 res )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;

	float3 point = rayInfo.pntList[ y * res.x +x] * rayInfo.voxel_size;
	if (point.x <= 0 && point.y <= 0 && point.z <= 0 ) return;
    
	int3 vox = make_int3(point);
	vox = vox - rayInfo.voxelCpyOffset;
	/*vox = rayInfo.voxelCpyDim;
	vox.x--;
	vox.y--;
	vox.z--;*/
	if (vox.x >= rayInfo.voxelCpyDim.x || vox.y >=rayInfo.voxelCpyDim.y || vox.z >= rayInfo.voxelCpyDim.z) return;
	atomicAdd(&rayInfo.voxelsCpy[vox.x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + vox.y * rayInfo.voxelCpyDim.z + vox.z], 1.0f);

	float *clrAddr = (float *)&rayInfo.voxelCpyClr[vox.x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + vox.y * rayInfo.voxelCpyDim.z + vox.z];
	uchar4 clr = ((uchar4 *)rayInfo.pntClrs)[ y * res.x +x];
	atomicAdd(clrAddr, float(clr.z) * 1.0f);
	atomicAdd(clrAddr+1, float(clr.y) * 1.0f);
	atomicAdd(clrAddr+2, float(clr.x) * 1.0f);
}

extern "C" __global__ void gvdbUpdateMapRaycast ( int3 res )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ( x >= res.x || y >= res.y ) return;

	//float3 jit = jitter_sample();
	float3 point = rayInfo.pntList[ y * res.x +x] * rayInfo.voxel_size;
	if (point.x <= 0 && point.y <= 0 && point.z <= 0) return;


	float3 ray = point - rayInfo.pos;
	float dist = length(ray);
	float3 sign = ray;
	sign.x = sign.x / abs(sign.x);
	sign.y = sign.y / abs(sign.y);
	sign.z = sign.z / abs(sign.z);

	float t = 0.0;
	float3 curPos = rayInfo.pos;
	int3 vox = make_int3(curPos) - rayInfo.voxelCpyOffset;
	float3 step;
	while (t < 1.0) {
		if (rayInfo.voxelsCpy[vox.x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + vox.y * rayInfo.voxelCpyDim.z + vox.z] <= 0) {
			atomicAdd(&rayInfo.voxelsCpy[vox.x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + vox.y * rayInfo.voxelCpyDim.z + vox.z], -1.0/100.0);
		}
		//step
		step.x = (floor(curPos.x + sign.x) - curPos.x) / ray.x;
		step.y = (floor(curPos.y + sign.y) - curPos.y) / ray.y;
		step.z = (floor(curPos.z + sign.z) - curPos.z) / ray.z;
		t += min(min(step.x, step.y), step.z);
		vox = make_int3(rayInfo.pos + t * ray) - rayInfo.voxelCpyOffset;
	}
}

extern "C" __global__ void gvdbUpdateMapVoxelCpy ( VDBInfo* gvdb, int3 atlasRes, uchar chan,float p1, float p2, float p3  )
{
	float3 wpos;
	GVDB_VOXUNPACKED

	if ( !getAtlasToWorld ( gvdb, atlasIdx, wpos )) return;
	wpos -= make_float3(0.5, 0.5, 0.5);
	if (wpos.x < 0 || wpos.y < 0 || wpos.z < 0 ) return;
	
	// is in region
	if (wpos.x < rayInfo.voxelCpyOffset.x || wpos.x >= rayInfo.voxelCpyOffset.x + rayInfo.voxelCpyDim.x || 
		wpos.y < rayInfo.voxelCpyOffset.y || wpos.y >= rayInfo.voxelCpyOffset.y + rayInfo.voxelCpyDim.y || 
		wpos.z < rayInfo.voxelCpyOffset.z || wpos.z >= rayInfo.voxelCpyOffset.z + rayInfo.voxelCpyDim.z ) return;

	int3 vox = make_int3(wpos);
	vox -= rayInfo.voxelCpyOffset;
	float v = surf3Dread<float>(gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	float update = rayInfo.voxelsCpy[vox.x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + vox.y * rayInfo.voxelCpyDim.z + vox.z];
	v += update;
	v = min(max(v, -20.0), 200000.0);
	surf3Dwrite( v, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	

	if ( update > 0.0f) {
		float3 newColor = rayInfo.voxelCpyClr[vox.x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + vox.y * rayInfo.voxelCpyDim.z + vox.z];

		uchar4 oldClr = surf3Dread<uchar4>(gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);
		newColor.x += float (oldClr.x);
		newColor.y += float (oldClr.y);
		newColor.z += float (oldClr.z);
		newColor /= (update+ 1.0f);
		uchar4 clr = make_uchar4(newColor.x, newColor.y, newColor.z, 0);
		surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);
		return;
	}
}

extern "C" __global__ void gvdbUpdateMapRegion ( VDBInfo* gvdb, int numPnts, int3 atlasRes )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= rayInfo.voxelCpyDim.x || y >= rayInfo.voxelCpyDim.y || z >= rayInfo.voxelCpyDim.z) return;

	float3 wpos = make_float3(float(x + rayInfo.voxelCpyOffset.x), float(y + rayInfo.voxelCpyOffset.y), float(z + rayInfo.voxelCpyOffset.z));
	if (wpos.x < 0 || wpos.y < 0 || wpos.z < 0 ) return;

	float3 offs, vmin; uint64 nid;
	VDBNode* node = getNodeAtPoint ( gvdb, wpos, &offs, &vmin, &nid );				// find vdb node at point
	if (node == 0x0) return;
	float3 pos = wpos - vmin;
	pos = make_float3(int(pos.x), int(pos.y), int(pos.z));
	if (pos.x < 0 || pos.y < 0 || pos.z < 0 || pos.x >= gvdb->res[0] || pos.y >= gvdb->res[0] || pos.z >= gvdb->res[0]) return;

	uint3 atlasIdx = make_uint3(offs) + make_uint3(pos);
	if (atlasIdx.x >= atlasRes.x || atlasIdx.y >= atlasRes.y || atlasIdx.z >= atlasRes.z) return;

	float v = surf3Dread<float>(gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	float update = rayInfo.voxelsCpy[x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + y * rayInfo.voxelCpyDim.z + z];
	v += update;
	v = min(max(v, -20.0), 200000.0);
	//printf("%ud, %ud, %ud", atlasIdx.x, atlasIdx.y, atlasIdx.z);
	surf3Dwrite( v, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	

	if ( update <= 0.0f) return;
	float3 newColor = rayInfo.voxelCpyClr[x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + y * rayInfo.voxelCpyDim.z + z];

	uchar4 oldClr = surf3Dread<uchar4>(gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);
	newColor.x += float(oldClr.x);
	newColor.y += float(oldClr.y);
	newColor.z += float(oldClr.z);
	newColor /= (update+ 1.0f);
	uchar4 clr = make_uchar4(newColor.x, newColor.y, newColor.z, 0);
	surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);
	return;
}

extern "C" __global__ void gvdbFillZero ( int3 res )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= rayInfo.voxelCpyDim.x || y >= rayInfo.voxelCpyDim.y || z >= rayInfo.voxelCpyDim.z) return;
	/*if (rayInfo.voxelsCpy[x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + y * rayInfo.voxelCpyDim.z + z] > 0) {//TODO: remove
		printf("%d, %d, %d\n", x,y,z);//TODO: remove
	}//TODO: remove*/

	if (res.x > 0) {//TODO: remove
		rayInfo.voxelsCpy[x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + y * rayInfo.voxelCpyDim.z + z] = 0.0f;
		rayInfo.voxelCpyClr[x * rayInfo.voxelCpyDim.y * rayInfo.voxelCpyDim.z + y * rayInfo.voxelCpyDim.z + z] = make_float3(0,0,0);
	}//TODO: remove
	return;
}

struct ALIGN(16) VirtualObjectData {
	uint8_t 	id;
	int 		numPnt;
	int 		numVol;
	int* 		lenVol;
	float3*		pntList;
};
__device__ VirtualObjectData		voInfo;

extern "C" __global__ void gvdbInsertVirtualObject ( VDBInfo* gvdb, int3 dimensions, int3 offset, int3 atlasRes )
{
	int volIdx, faceIdx;
	int pointPos = 0;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= dimensions.x || y >= dimensions.y || z >= dimensions.z) return;

	float3 wpos = make_float3(float(x + offset.x), float(y + offset.y), float(z + offset.z));
	if (wpos.x < 0 || wpos.y < 0 || wpos.z < 0 ) return;

	for (volIdx = 0; volIdx < voInfo.numVol; volIdx++) {
		bool insert = true;
		for (faceIdx = 0; faceIdx < voInfo.lenVol[volIdx]; faceIdx++) {
			float3 position = voInfo.pntList[pointPos + faceIdx * 2];
			float3 normal = voInfo.pntList[pointPos + faceIdx * 2 + 1];
			float dist = dot(wpos - position, normal);
			if (x == 25 && y == 25 && z == 25) {
			}
			if (dist > 0) insert = false;
		}
		if (insert) {
			float3 offs, vmin; uint64 nid;
			VDBNode* node = getNodeAtPoint ( gvdb, wpos, &offs, &vmin, &nid );				// find vdb node at point
			if (node == 0x0) return;
			float3 pos = wpos - vmin;
			pos = make_float3(int(pos.x), int(pos.y), int(pos.z));
			if (pos.x < 0 || pos.y < 0 || pos.z < 0 || pos.x >= gvdb->res[0] || pos.y >= gvdb->res[0] || pos.z >= gvdb->res[0]) return;

			uint3 atlasIdx = make_uint3(offs) + make_uint3(pos);
			if (atlasIdx.x >= atlasRes.x || atlasIdx.y >= atlasRes.y || atlasIdx.z >= atlasRes.z) return;

			float v = 100000;
			surf3Dwrite( v, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);

			uchar4 clr = make_uchar4(0, 0, 255, voInfo.id);
			surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);
			return;
		}
		pointPos += voInfo.lenVol[volIdx] * 2;
	}

	
	return;
}

/*	
FOR DEBUGGING
	# if __CUDA_ARCH__>=200
    printf("%f, %f, %f \n", wpos.x, wpos.y, wpos.z);
    printf("%f, %f, %f \n\n", frame.pos.x, frame.pos.y, frame.pos.z);

	#endif 
	*/