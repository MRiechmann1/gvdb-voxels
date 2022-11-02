struct ALIGN(16) FrameInfo {
	float3*		pntList;
	uint*		pntClrs;
	int3		gridRes;
	float3		gridSize;	
	float3		cams;
	float3		camu;
	float3		camv;
	float3 		pos;
	float  		maxDist; // in voxels number
	float  		minDist; // in voxels number
};
__device__ FrameInfo		frame;

extern "C" __global__ void gvdbUpdateMap ( VDBInfo* gvdb, int3 atlasRes, uchar chan,float p1, float p2, float p3  )
{
	float3 relPos, wpos, wnorm, xRef, yRef;
	int xmin, xmax, ymin, ymax;
	float dotX, dotY, dotGlobal;
	float len;
	GVDB_VOXUNPACKED

	if ( !getAtlasToWorld ( gvdb, atlasIdx, wpos )) return;


	/*
	 * Check if voxel is in range bounds
	 */
	relPos = wpos - frame.pos; //  pos wrong? get relative position
	len = length(relPos);
	if (len >  frame.maxDist || len < frame.minDist) return; //len in voxel size // len not in estmiated max and min distance
 


	uchar4 clr = make_uchar4(255, 125, 125, 255);
	surf3Dwrite( clr, gvdb->volOut[1], atlasIdx.x * sizeof(uchar4), atlasIdx.y, atlasIdx.z);

	float prob = 6; // setVolumeRange setzt -1 als max (nur danm gerendert) anpassen der von setVolumeRange und renderPipeline
	surf3Dwrite( prob, gvdb->volOut[0], atlasIdx.x * sizeof(float), atlasIdx.y, atlasIdx.z);
	return;

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