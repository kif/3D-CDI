/* 2D kernel, one thread per input pixel
 * 
 * pixel start at 0 and finish at 1, the center is at 0.5
 * thread ids follow the memory location convention (zyx) not the math x,y,z convention 
 */ 

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

float2 inline calc_position_real(float2 index,
                                 float2 center,
                                 float pixel_size)
{
    return (index - center) * pixel_size;
}


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
    float scale = distance * distance / pixel_size; 
    return scale*(float3)(dot(Rx, pos3), dot(Ry, pos3), dot(Rz, pos3));
}
                                

kernel void regid_CDI(global float* image,
                      const  int height,
                      const  int width,
                      const  float pixel_size,
                      const  float distance,
                      const  float phi,
                      const  float center_x,
                      const  float center_y,
                      global float *volume,
                      const  int volume_shape,
                      const  int oversampling)
{
    int tmp;
    size_t where_in, where_out;
    float value, cos_phi, sin_phi;
    float2 pos2, center = (float2)(center_x, center_y);
    float3 Rx, Ry, Rz, recip;
    //float4 corners_x, corners_y;
    
    if ((get_global_id(0)>=height) || (get_global_id(1)>=width))
        return;
    
    where_in = width*get_global_id(0)+get_global_id(1);
    
    cos_phi = cos(phi*M_PI_F/180.0f);
    sin_phi = sin(phi*M_PI_F/180.0f);
    Rx = (float3)(cos_phi, 0.0f, sin_phi);
    Ry = (float3)(0.0f, 1.0f, 0.0f);
    Rz = (float3)(-sin_phi, 0.0f, cos_phi);
    
    pos2 = (float2)(get_global_id(1)+0.5f, get_global_id(1) + 0.5f); //this is the center of the pixel
    recip = calc_position_rec(pos2, center, pixel_size, distance, Rx, Ry, Rz);
    
    value = image[where_in];

    // No oversampling for now
    
    tmp = (int)recip.x + volume_shape/2;
    if ((tmp>=0) && (tmp<volume_shape))
    {
        where_out = tmp;
        tmp = (int)recip.y + volume_shape/2;
        if ((tmp>=0) && (tmp<volume_shape))
        {
            where_out += tmp*volume_shape;
            tmp = (int)recip.z + volume_shape/2;
            if ((tmp>=0) && (tmp<volume_shape))
            {
                where_out += tmp*volume_shape*volume_shape;    
                atomic_add_global_float(&volume[2*where_out], value);
                atomic_add_global_float(&volume[2*where_out+1], 1.0f);
            }
        }               
    }
}

                              
                             
                             
                               