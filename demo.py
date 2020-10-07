#!/usr/bin/env python3

"""
3D CDI preprocessing 

Regrid frames into 3D regular Fourier space
"""

import numpy

nframes = 512
shape = 512, 512
volume = (512, 512, 512)
center = (260, 250)
pixel_size = 55e-6
distance = 1

phi = numpy.linspace(-80, 80, nframes)

def calc_coord(phi, distance, pixel_size, shape, center):
    Y, Y numpy.ogrid[:shape[0], :shape[1]]
                     
    xyz = 

