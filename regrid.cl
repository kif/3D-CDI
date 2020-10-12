//Storage for that many voxel per pixel
#define STORAGE_SIZE 64

// Function to perform an atom addition in global memory (does not exist in OpenCL)
inline void atomic_add_global_float(volatile global float *addr, float val)
{
   union {
       uint  u32;
       float f32;
   } next, expected, current;
   current.f32    = *addr;
   do {
       expected.f32 = current.f32;
       next.f32     = expected.f32 + val;
       current.u32  = atomic_cmpxchg( (volatile global uint *)addr,
                                      expected.u32, next.u32);
   } while( current.u32 != expected.u32 );
}

// Performs the centering/scaling in the detector plane. Image flipping must be implemented here
float2 inline calc_position_real(float2 index,
                                 float2 center,
                                 float pixel_size)
{
    return (index - center) * pixel_size;
}

// Transforms a 2D position in the image into a 3D coordinate in the volume
float3 inline calc_position_rec(float2 index,
                                float2 center,
                                float pixel_size,
                                float distance,
                                float3 Rx,
                                float3 Ry,
                                float3 Rz)
{
    float2 pos2 = calc_position_real(index, center, pixel_size);
    float d = sqrt(distance*distance + dot(pos2, pos2));
    float3 pos3 = (float3)(pos2.x/d, pos2.y/d, distance/d-1.0f);
    float scale = distance*distance/pixel_size;
    return scale * (float3)(dot(Rx, pos3), dot(Ry, pos3), dot(Rz, pos3));
}

/* Performs the regridding of an image on a 3D volume
 *
 * 2D kernel, one thread per input pixel. Scatter-like kernel with atomics.
 * 
 * pixel start at 0 and finish at 1, the center is at 0.5
 * thread ids follow the memory location convention (zyx) not the math x,y,z convention 
 *   
 * Basic oversampling implemented but slows down the processing, mainly for calculating 
 * Atomic operations are the second bottleneck
 */ 

    
kernel void regid_CDI_simple(global float* image,
                             const  int    height,
                             const  int    width,
                             const  float  pixel_size,
                             const  float  distance,
                             const  float  phi,
                             const  float  center_x,
                             const  float  center_y,
                             global float* signal,
                             global float* norm,
                             const  int    shape,
                                    int    oversampling)
{
    int tmp, shape_2, i, j, k;
    size_t where_in, where_out;
    float value, cos_phi, sin_phi, delta, start;
    float2 pos2, center = (float2)(center_x, center_y);
    float3 Rx, Ry, Rz, recip;
        
    if ((get_global_id(0)>=height) || (get_global_id(1)>=width))
        return;
    
    where_in = width*get_global_id(0)+get_global_id(1);
    shape_2 = shape/2;
    oversampling = (oversampling<1?1:oversampling);
    start = 0.5f / oversampling;
    delta = 2 * start;
    
    cos_phi = cos(phi*M_PI_F/180.0f);
    sin_phi = sin(phi*M_PI_F/180.0f);
    Rx = (float3)(cos_phi, 0.0f, sin_phi);
    Ry = (float3)(0.0f, 1.0f, 0.0f);
    Rz = (float3)(-sin_phi, 0.0f, cos_phi);
    
    // No oversampling for now
    //this is the center of the pixel
    //pos2 = (float2)(get_global_id(1)+0.5f, get_global_id(0) + 0.5f); 
    
    //Basic oversampling    

    for (i=0; i<oversampling; i++)
    {
        for (j=0; j<oversampling; j++)
        {
            pos2 = (float2)(get_global_id(1) + start + i*delta, 
                            get_global_id(0) + start + j*delta); 
            recip = calc_position_rec(pos2, center, pixel_size, distance, Rx, Ry, Rz);
            value = image[where_in];
    
            tmp = (int)recip.x + shape/2;
            if ((tmp>=0) && (tmp<shape))
            {
                where_out = tmp;
                tmp = (int)recip.y + shape_2;
                if ((tmp>=0) && (tmp<shape))
                {
                    where_out += tmp * shape;
                    tmp = (int)recip.z + shape_2;
                    if ((tmp>=0) && (tmp<shape))
                    {
                        where_out += ((long)tmp) * shape * shape;  
                        atomic_add_global_float(&signal[where_out], value);
                        atomic_add_global_float(&norm[where_out], 1.0f);
                    }
                }               
            }            
        }
    }
}

kernel void regid_CDI(global float* image,
                      const  int    height,
                      const  int    width,
                      const  float  dummy,
                      const  float  pixel_size,
                      const  float  distance,
                      const  float  phi,
                      const  float  center_x,
                      const  float  center_y,
                      global float* signal,
                      global int*   norm,
                      const  int    shape,
                      int    oversampling)
{
    int tmp, shape_2, i, j, k;
    size_t where_in, where_out;
    float value, delta;
    float2 pos2, center = (float2)(center_x, center_y);
    float3 Rx, Ry, Rz, recip;
    
    //This is local storage of voxels to be written
    int last=0;
    size_t index[STORAGE_SIZE];
    float2 store[STORAGE_SIZE];
    
    
    
    if ((get_global_id(0)>=height) || (get_global_id(1)>=width))
        return;
    
    where_in = width*get_global_id(0)+get_global_id(1);
    shape_2 = shape/2;
    oversampling = (oversampling<1?1:oversampling);
    delta = 1.0f / oversampling;
    {   
        float cos_phi, sin_phi;
        cos_phi = cos(phi*M_PI_F/180.0f);
        sin_phi = sin(phi*M_PI_F/180.0f);
        Rx = (float3)(cos_phi, 0.0f, sin_phi);
        Ry = (float3)(0.0f, 1.0f, 0.0f);
        Rz = (float3)(-sin_phi, 0.0f, cos_phi);
        
    }
    
    // No oversampling for now
    //this is the center of the pixel
    //pos2 = (float2)(get_global_id(1)+0.5f, get_global_id(0) + 0.5f); 
    
    
    value = image[where_in];
    
    //dynamic masking
    if (value == dummy) return;
    
    //Basic oversampling    
    for (i=0; i<oversampling; i++)
    {
        for (j=0; j<oversampling; j++)
        {
            pos2 = (float2)(get_global_id(1) + (i + 0.5f)*delta, 
                            get_global_id(0) + (j + 0.5f)*delta); 
            recip = calc_position_rec(pos2, center, pixel_size, distance, Rx, Ry, Rz);
            
    
            tmp = (int)recip.x + shape/2;
            if ((tmp>=0) && (tmp<shape))
            {
                where_out = tmp;
                tmp = (int)recip.y + shape_2;
                if ((tmp>=0) && (tmp<shape))
                {
                    where_out += tmp * shape;
                    tmp = (int)recip.z + shape_2;
                    if ((tmp>=0) && (tmp<shape))
                    {
                        where_out += ((long)tmp) * shape * shape;                          
                        signal[where_out] += value;
                        norm[where_out] += 1.0f;
                        
                        //storage locally
                        int found = 0;
                        for (k=0; k<last; k++)
                        {
                            if (where_out == index[k])
                            {
                                    store[k] += (float2)(value, 1.0f);
                                    found = 1;
                                    k = last;
                            }
                        }
                        if (found == 0)
                        {
                            if (last >= STORAGE_SIZE)
                                printf("Too many voxels covered by pixel\n");
                            else
                            {
                                index[last] = where_out;
                                store[last] = (float2)(value, 1.0f);
                                last++;
                            }
                        }  
                    }
                }               
            }            
        }
    }
    // Finally we update the global memory with atomic writes
    for (k=0; k<last; k++)
    {
        atomic_add_global_float(&signal[index[k]], store[k].s0);
        atomic_add(&norm[index[k]], (int)store[k].s1);
    }
}
