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

	minV.y = min(koef[0].y, koef[1].y);
	minV.y = min(minV.y, koef[2].y);
	minV.y = min(minV.y, koef[3].y);
	minV.y = min(minV.y, koef[4].y);
	minV.y = min(minV.y, koef[5].y);
	minV.y = min(minV.y, koef[6].y);
	minV.y = min(minV.y, koef[7].y);
	minV.y = max(minV.y, 0.0);

	minV.z = min(koef[0].z, koef[1].z);
	minV.z = min(minV.z, koef[2].z);
	minV.z = min(minV.z, koef[3].z);
	minV.z = min(minV.z, koef[4].z);
	minV.z = min(minV.z, koef[5].z);
	minV.z = min(minV.z, koef[6].z);
	minV.z = min(minV.z, koef[7].z);
	minV.z = max(minV.z, 0.0);

	return minV;
}

float3 __device__ __inline__ maxValue(float3 *koef) {
	float3 minV;
	/*minV.x = max(koef[0].x, koef[1].x);
	minV.x = max(minV.x, koef[2].x);
	minV.x = max(minV.x, koef[3].x);
	minV.x = max(minV.x, koef[4].x);
	minV.x = max(minV.x, koef[5].x);
	minV.x = max(minV.x, koef[6].x);
	minV.x = max(minV.x, koef[7].x);
	minV.x = min(minV.x, 1.0);*/

	minV.y = max(koef[0].y / koef[0].x, koef[1].y / koef[1].x);
	minV.y = max(minV.y, koef[2].y / koef[2].x);
	minV.y = max(minV.y, koef[3].y / koef[3].x);
	minV.y = max(minV.y, koef[4].y / koef[4].x);
	minV.y = max(minV.y, koef[5].y / koef[5].x);
	minV.y = max(minV.y, koef[6].y / koef[6].x);
	minV.y = max(minV.y, koef[7].y / koef[7].x);
	minV.y = min(minV.y, 1.0);

	minV.z = max(koef[0].z / koef[0].x, koef[1].z / koef[1].x);
	minV.z = max(minV.z, koef[2].z / koef[2].x);
	minV.z = max(minV.z, koef[3].z / koef[3].x);
	minV.z = max(minV.z, koef[4].z / koef[4].x);
	minV.z = max(minV.z, koef[5].z / koef[5].x);
	minV.z = max(minV.z, koef[6].z / koef[6].x);
	minV.z = max(minV.z, koef[7].z / koef[7].x);
	minV.z = min(minV.z, 1.0);

	return minV;
}

extern "C" __global__ void gvdbUpdateMap ( VDBInfo* gvdb, int3 atlasRes, uchar chan,float p1, float p2, float p3  )
{
	float3 relPos, wpos, wnorm, xRef, yRef, koef[8];
	int xmin, xmax, ymin, ymax;
	float dotX, dotY, dotGlobal;
	float len;
	GVDB_VOXUNPACKED

	if ( !getAtlasToWorld ( gvdb, atlasIdx, wpos )) return;


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

		float prob = max(v - hitCount, frame.minProb);//6;//max(v - hitCount, frame.minProb);
		return;
	}
	return;
	/*
		uchar4 clr = ((uchar4 *)frame.pntClrs)[i];
		surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);

				float prob = 6; // setVolumeRange setzt -1 als max (nur danm gerendert) anpassen der von setVolumeRange und renderPipeline
				surf3Dwrite( prob, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);*/

	/*if (frame.pntList[21600].x <= wpos.x || frame.pntList[21600].x > wpos.x+1 ||
		frame.pntList[21600].y <= wpos.y || frame.pntList[21600].y > wpos.y+1 ||
		frame.pntList[21600].y <= wpos.y || frame.pntList[21600].y > wpos.y+1) return;*/
	/*float3 centerRay = 0.5*scan.camu + 0.5*scan.camv + scan.cams;
	float3 hitPos = pos + tnearest.x * dir;
	float n = dot(hitPos - pos, normal) / dot(normal, normal);	*/
	
	
	
	/*for (uint i = 0; i < frame.numPts; i++) {
		if (frame.pntList[i].x >= wpos.x && frame.pntList[i].x < wpos.x+1 &&
		frame.pntList[i].y >= wpos.y && frame.pntList[i].y < wpos.y+1 &&
		frame.pntList[i].z >= wpos.z && frame.pntList[i].z < wpos.z+1) {
		//if (intersection(wpos, frame.pos, frame.pntList[i])) {
			uchar4 clr = make_uchar4(255, 125, 125, 255);
			surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);

			float prob = 6; // setVolumeRange setzt -1 als max (nur danm gerendert) anpassen der von setVolumeRange und renderPipeline
			surf3Dwrite( prob, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
		}
	}

	return;*/

	// write color


	/*
	 * Check if voxel is in fov
	 */	
	//wnorm = wpos / len;
	//dotGlobal = dot(wnorm, cameraZ);
	//if (dotGlobal < 0.3420214) return; 	// 0.3420214 is equivalent to 70 degree
										// a value smaller means the angle is greater than 70 degree and therefor out of fov

	/*
	 * Find corresponding rays
	 */
	//xRef = dot(cameraY, wnorm);
	//yRef = dot(cameraX, wnorm);*/

}

// Follow the implementation of scanBuilding (especially raxbox intersect), to implemnt ray casting based insertion
// Voxel based implementation see board

/*	
FOR DEBUGGING
	# if __CUDA_ARCH__>=200
    printf("%f, %f, %f \n", wpos.x, wpos.y, wpos.z);
    printf("%f, %f, %f \n\n", frame.pos.x, frame.pos.y, frame.pos.z);

	#endif 
	*/