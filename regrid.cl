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
    // float d = sqrt(distance*distance + dot(pos2, pos2));
    float d = fast_length((float3)(distance, pos2));
    float3 pos3 = (float3)(pos2.x/d, pos2.y/d, distance/d-1.0f);
    float scale = distance/pixel_size;
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
    
            tmp = convert_int_rtn(recip.x) + shape_2;
            if ((tmp>=0) && (tmp<shape))
            {
                where_out = tmp;
                tmp = convert_int_rtn(recip.y) + shape_2;
                if ((tmp>=0) && (tmp<shape))
                {
                    where_out += tmp * shape;
                    tmp = convert_int_rtn(recip.z) + shape_2;
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

/*
 * Regrid an image (heightxwidth) to a 3D volume (shape³) 
 * 
 * 
 * 
 */

kernel void regid_CDI(global float* image,
                      global uchar* mask,
                      const  int    height,
                      const  int    width,
                      const  float  pixel_size,
                      const  float  distance,
                      const  float  phi,
                             float  dphi,
                      const  float  center_x,
                      const  float  center_y,
                      global float* signal,
                      global int*   norm,
                      const  int    shape,
                             int    oversampling_pixel,
                             int    oversampling_phi)
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
    
    where_in = width*get_global_id(0)+get_global_id(1);
    shape_2 = shape/2;
    oversampling_pixel = (oversampling_pixel<1?1:oversampling_pixel);
    oversampling_phi = (oversampling_phi<1?1:oversampling_phi);
    delta = 1.0f / oversampling_pixel;
    dphi /= oversampling_phi;
    
    { //Manual mask definition
        int y = get_global_id(0),
            x = get_global_id(1);
        if ((x >= width) ||
            (y >= height))
            return;
    }
    { // static mask
        if (mask[where_in])
            return;
    }
    {//dynamic masking
        value = image[where_in];
        if (value < -10.0f)
            return;
        else if (value <=0.0f)
            value= 0.0f;
        if (! isfinite(value)) 
            return;
    }
    

    
    // No oversampling for now
    //this is the center of the pixel
    //pos2 = (float2)(get_global_id(1)+0.5f, get_global_id(0) + 0.5f); 

    
    //Basic oversampling
    for (int dr=0; dr<oversampling_phi; dr++)
    {
        float cos_phi, sin_phi, rphi;
        rphi = (phi + (0.0f + dr)*dphi) * M_PI_F/180.0f; 
        cos_phi = cos(rphi);
        sin_phi = sin(rphi);
        Rx = (float3)(cos_phi, 0.0f, sin_phi);
        Ry = (float3)(0.0f, 1.0f, 0.0f);
        Rz = (float3)(-sin_phi, 0.0f, cos_phi);
        for (i=0; i<oversampling_pixel; i++)
        {
            for (j=0; j<oversampling_pixel; j++)
            {
                pos2 = (float2)(get_global_id(1) + (i + 0.5f)*delta, 
                                get_global_id(0) + (j + 0.5f)*delta); 
                recip = calc_position_rec(pos2, center, pixel_size, distance, Rx, Ry, Rz);
                
                tmp = convert_int_rtn(recip.x) + shape_2;
                if ((tmp>=0) && (tmp<shape))
                {
                    where_out = tmp;
                    tmp = convert_int_rtn(recip.y) + shape_2;
                    if ((tmp>=0) && (tmp<shape))
                    {
                        where_out += tmp * shape;
                        tmp = convert_int_rtn(recip.z) + shape_2;
                        if ((tmp>=0) && (tmp<shape))
                        {
                            where_out += ((long)tmp) * shape * shape;                          
                            
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
    }
    // Finally we update the global memory with atomic writes
    for (k=0; k<last; k++)
    {
        atomic_add_global_float(&signal[index[k]], store[k].s0);
        atomic_add(&norm[index[k]], (int)store[k].s1);
        //signal[index[k]] += store[k].s0;
        //norm[index[k]] += (int)store[k].s1;
    }
}

/*
 * Regrid an image (heightxwidth) to a 3D volume (shape³) 
 * 
 * Slabed, store only data starting at slab_start <= z < slab_end
 * 
 */

kernel void regid_CDI_slab(global float* image,
                           global uchar* mask,
                           const  int    height,
                           const  int    width,
                           const  float  pixel_size,
                           const  float  distance,
                           const  float  phi,
                                  float  dphi,
                           const  float  center_x,
                           const  float  center_y,
                           global float* signal,
                           global int*   norm,
                           const  int    shape,
                           const  int    slab_start,
                           const  int    slab_end,
                                  int    oversampling_pixel,
                                  int    oversampling_phi)
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
    
    where_in = width*get_global_id(0)+get_global_id(1);
    shape_2 = shape/2;
    oversampling_pixel = (oversampling_pixel<1?1:oversampling_pixel);
    oversampling_phi = (oversampling_phi<1?1:oversampling_phi);
    delta = 1.0f / oversampling_pixel;
    dphi /= oversampling_phi;
    
    { //Manual mask definition
        int y = get_global_id(0),
            x = get_global_id(1);
        if ((x >= width) ||
            (y >= height))
            return;
    }
    { // static mask
        if (mask[where_in])
            return;
    }
    {//dynamic masking
        value = image[where_in];
        if (value < -10.0f)
            return;
        else if (value <=0.0f)
            value= 0.0f;
        if (! isfinite(value)) 
            return;
    }
    

    
    // No oversampling for now
    //this is the center of the pixel
    //pos2 = (float2)(get_global_id(1)+0.5f, get_global_id(0) + 0.5f); 

    
    //Basic oversampling
    for (int dr=0; dr<oversampling_phi; dr++)
    {
        float cos_phi, sin_phi, rphi;
        rphi = (phi + (0.0f + dr)*dphi) * M_PI_F/180.0f; 
        cos_phi = cos(rphi);
        sin_phi = sin(rphi);
        Rx = (float3)(cos_phi, 0.0f, sin_phi);
        Ry = (float3)(0.0f, 1.0f, 0.0f);
        Rz = (float3)(-sin_phi, 0.0f, cos_phi);
        for (i=0; i<oversampling_pixel; i++)
        {
            for (j=0; j<oversampling_pixel; j++)
            {
                pos2 = (float2)(get_global_id(1) + (i + 0.5f)*delta, 
                                get_global_id(0) + (j + 0.5f)*delta); 
                recip = calc_position_rec(pos2, center, pixel_size, distance, Rx, Ry, Rz);
                
                tmp = convert_int_rtn(recip.x) + shape_2;
                if ((tmp>=0) && (tmp<shape))
                {
                    where_out = tmp;
                    tmp = convert_int_rtn(recip.y) + shape_2;
                    if ((tmp>=0) && (tmp<shape))
                    {
                        where_out += tmp * shape;
                        tmp = convert_int_rtn(recip.z) + shape_2;
                        if ((tmp>=slab_start) && (tmp<slab_end))
                        {
                            where_out += ((long)(tmp-slab_start)) * shape * shape;                          
                            
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
    }
    // Finally we update the global memory with atomic writes
    for (k=0; k<last; k++)
    {
        atomic_add_global_float(&signal[index[k]], store[k].s0);
        atomic_add(&norm[index[k]], (int)store[k].s1);
        //signal[index[k]] += store[k].s0;
        //norm[index[k]] += (int)store[k].s1;
    }
}
