#!/usr/bin/env python3

"""
3D CDI preprocessing

Regrid frames into 3D regular Fourier space
"""

import numpy
import pyopencl as cl
from pyopencl import array as cla
import time
import glob
import fabio
import h5py, hdf5plugin
from pyFAI.utils.shell import ProgressBar

frames = glob.glob('/mnt/data/ID10/CDI/SiO2msgel3_cand1/img_*.edf')

nframes = len(frames)
shape = numpy.int32(568), numpy.int32(568)
volume = (numpy.int32(568), numpy.int32(568), numpy.int32(568))
center = (numpy.float32(284), numpy.float32(284))
pixel_size = numpy.float32(55e-6)
distance = numpy.float32(3.3)
oversampling = numpy.int32(8)
dphi = numpy.float32(0.2)
ldphi = numpy.linspace(0, 0.2, oversampling, endpoint=False, dtype=numpy.float32)
dummy = numpy.float32(0.0)

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

print(nframes, shape, volume, oversampling)
print(ctx)

with open("regrid.cl", "r") as f:
    kernel_src = f.read()

prg = cl.Program(ctx, kernel_src).build()

image_d = cla.empty(queue, shape, dtype=numpy.float32)
signal_d = cla.empty(queue, volume, dtype=numpy.float32)
norm_d = cla.empty(queue, volume, dtype=numpy.int32)
mask_d = cla.empty(queue, shape, dtype=numpy.uint8)
mask_d.set(fabio.open("mask_ID10.edf").data)


def meas_phi(f):
    return numpy.float32(f.header.get("motor_pos").split()[f.header.get("motor_mne").split().index("ths")])


signal_d.fill(0.0)
norm_d.fill(0)
ws = (8, 4)

pb = ProgressBar("Projecting frames", nframes, 30)
t0 = time.perf_counter()
for j, i in enumerate(frames):
    f = fabio.open(i)
    image_d.set(f.data)
    phi = meas_phi(f)
#     for dphi in ldphi:
    evt = prg.regid_CDI(queue, shape, ws,
                            image_d.data,
                            mask_d.data,
                            * shape,
                            dummy,
                            pixel_size,
                            distance,
                            phi, dphi,
                            *center,
                            signal_d.data,
                            norm_d.data,
                            volume[0],
                            oversampling)
    pb.update(j)
evt.wait()
try:
    volume_h = (signal_d / norm_d).get()
except cl._cl.MemoryError:
    volume_h = signal_d.get() / norm_d.get().astype(numpy.float32)

t1 = time.perf_counter()
print("Execution time: ", t1 - t0, "s")

with h5py.File("regrid_mask.h5", mode="w") as h:
    h.create_dataset("SiO2msgel3",
            data=numpy.ascontiguousarray(volume_h, dtype=numpy.float32),
            # **hdf5plugin.Zfp(reversible=True))
            ** hdf5plugin.Bitshuffle())
    h["oversampling"] = oversampling
    h["kernel"] = kernel_src
