#!/usr/bin/env python3

"""
3D CDI preprocessing 

Regrid frames into 3D regular Fourier space
"""

import numpy
import pyopencl as cl
from pyopencl import array as cla
import time

nframes = 512
shape = numpy.int32(512), numpy.int32(512)
volume = (numpy.int32(512), numpy.int32(512), numpy.int32(512))
center = (numpy.float32(260), numpy.float32(250))
pixel_size = numpy.float32(55e-6)
distance = numpy.float32(3)
phi = numpy.linspace(-80, 80, nframes).astype(numpy.float32)
oversampling = numpy.int32(16)

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

print(nframes, shape, volume)
print(ctx)

with open("regrid.cl", "r") as f:
    prg = cl.Program(ctx, f.read()).build()

image_d = cla.empty(queue, shape, dtype=numpy.float32)
signal_d = cla.empty(queue, volume, dtype=numpy.float32)
norm_d = cla.empty(queue, volume, dtype=numpy.float32)
# ws = (32,32)
# shared = cl.LocalMemory(4*4*(ws[0]+1)*(ws[1]+1))

signal_d.fill(0.0)
norm_d.fill(0.0)
image_d.fill(1.0)
ws = (32, 32)

t0 = time.perf_counter()
for i in phi:
    evt = prg.regid_CDI(queue, shape, ws,
                            image_d.data,
                            *shape,
                            pixel_size,
                            distance,
                            i,
                            *center,
                            signal_d.data,
                            norm_d.data,
                            volume[0],
                            oversampling)
evt.wait()
t1 = time.perf_counter()
print("Execution time: ", t1 - t0)
print(signal_d.get().mean(), norm_d.get().mean())
