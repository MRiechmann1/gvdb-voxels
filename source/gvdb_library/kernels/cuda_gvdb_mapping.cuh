extern "C" __global__ void gvdbUpdateMap ( VDBInfo* gvdb, int3 atlasRes, uchar chan,float p1, float p2, float p3  )
{
	/*float3 wpos, wnorm, xRef, yRef;
	int xmin, xmax, ymin, ymax;
	float dotX, dotY, dotGlobal;
	float len;*/
	GVDB_VOXUNPACKED
    uchar4 v;// = 255 << 24 || 255 << 16 || 255 << 8 || 255;
	v.x = 255;
	v.y = 125;
	v.z = 125;
	v.w = 255;
	surf3Dwrite( v, gvdb->volOut[chan], atlasIdx.x * sizeof(int), atlasIdx.y, atlasIdx.z);
	
	/*if ( !getAtlasToWorld ( gvdb, vox, wpos )) return;
	
	/*
	 * Check if voxel is in range bounds
	 */
	//wpos = wpos - cameraPos; // get relative position
	//len = lenght(wpos);
	//if (len > 4.0 || len < 0.15) return; // len not in estmiated max and min distance

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