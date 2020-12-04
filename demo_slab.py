#!/usr/bin/env python3

"""
3D CDI preprocessing

Regrid frames into 3D regular Fourier space
"""
from math import ceil
import numpy
import pyopencl as cl
from pyopencl import array as cla
import time
import glob
import fabio
import h5py, hdf5plugin
from pyFAI.utils.shell import ProgressBar

frames = {i:None for i in glob.glob('/mnt/data/ID10/CDI/SiO2msgel3_cand1/img_*.edf')}

nframes = len(frames)
shape = numpy.int32(568), numpy.int32(568)
volume = (numpy.int32(568), numpy.int32(568), numpy.int32(568))
center = (numpy.float32(284), numpy.float32(284))
pixel_size = numpy.float32(55e-6)
distance = numpy.float32(3.3)
oversampling = numpy.int32(8)
oversampling_phi = numpy.int32(8)
dphi = numpy.float32(0.2)

nb_slab = 32
slab_heigth = numpy.int32((volume[0] / nb_slab) + 1)

ws = (8, 4)
bs = [int(ceil(s / w) * w) for s, w in zip(shape, ws)]

ctx = cl.create_some_context(interactive=False)
queue = cl.CommandQueue(ctx)

print(f"{nframes} frames of {shape}, projected in a volume of {volume}, with an oversampling of {oversampling_phi}-{oversampling}-{oversampling}.")
print(f"Working on device {ctx.devices[0].name}")

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


pb = ProgressBar("Projecting frames", nframes * nb_slab, 30)
jj = 0
t0 = time.perf_counter()

with h5py.File(f"regrid_slab-{oversampling_phi}-{oversampling}-{oversampling}.h5", mode="w") as h:
    dataset = h.create_dataset("SiO2msgel3",
                              shape=volume,
                              dtype=numpy.float32,
                              chunks=(slab_heigth,) + volume[1:],
                              ** hdf5plugin.Bitshuffle())
    h["oversampling_pixel"] = oversampling
    h["oversampling_phi"] = oversampling_phi
    h["kernel"] = kernel_src

    for slab_start in numpy.arange(0, volume[0], slab_heigth, dtype=numpy.int32):
        slab_end = min(slab_start + slab_heigth, volume[0])
        signal_d.fill(0.0)
        norm_d.fill(0)

        for j, i in enumerate(frames):
            f = frames[i]
            if f is None:
                f = frames[i] = fabio.open(i)
            image_d.set(f.data)
            phi = meas_phi(f)
            pb.update(jj, f"Project frame #{j} onto slab [{slab_start}:{slab_end}]")
            evt = prg.regid_CDI_slab(queue, bs, ws,
                                    image_d.data,
                                    mask_d.data,
                                    * shape,
                                    pixel_size,
                                    distance,
                                    phi, dphi,
                                    *center,
                                    signal_d.data,
                                    norm_d.data,
                                    volume[0],
                                    slab_start,
                                    slab_end,
                                    oversampling,
                                    oversampling_phi)

            jj += 1
        pb.update(jj, "Save to HDF5")
        volume_h = signal_d.get() / norm_d.get().astype(numpy.float32)
        dataset[slab_start:slab_end] = volume_h[:slab_end - slab_start]
t1 = time.perf_counter()
print(f"\nExecution time: {t1 - t0} s")

