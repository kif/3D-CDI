#!/usr/bin/env python3
# coding: utf-8

"""
Rebuild the 3D reciprocal space  
by projecting a set of 2d speckle SAXS pattern taken at various rotation angles 
into a 3D regular volume
"""

__author__ = "Jérôme Kieffer"
__copyright__ = "2020 ESRF"
__license__ = "MIT"
__version__ = "0.1"
__date__ = "04/12/2020"

import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from math import ceil
import numpy
import pyopencl as cl
from pyopencl import array as cla
import time
import glob
import fabio
import h5py
import hdf5plugin
from pyFAI.utils.shell import ProgressBar
from silx.opencl.processing import OpenclProcessing
from silx.opencl.common import query_kernel_info
import argparse

logger = logging.getLogger("preprocess_cdi")

EXIT_SUCCESS = 0
EXIT_FAILURE = 1
EXIT_ARGUMENT_FAILURE = 2


def as_str(smth):
    "Ensure to be a string"
    if isinstance(smth, bytes):
        return smth.decode()
    else:
        return str(smth)


def expand_args(args):
    """
    Takes an argv and expand it (under Windows, cmd does not convert *.tif into
    a list of files.

    :param list args: list of files or wildcards
    :return: list of actual args
    """
    new = []
    for afile in args:
        if glob.has_magic(afile):
            new += glob.glob(afile)
        else:
            new.append(afile)
    return new


def parse():

    epilog = """Assumption: There is enough memory to hold all frames in memory
     
                return codes: 0 means a success. 1 means the conversion
                contains a failure, 2 means there was an error in the
                arguments"""

    parser = argparse.ArgumentParser(prog="cdi-regrid",
                                     description=__doc__,
                                     epilog=epilog)
    parser.add_argument("IMAGE", nargs="*",
                        help="file with input images in Bliss format HDF5")
    parser.add_argument("-V", "--version", action='version', version=__date__,
                        help="output version and exit")
    parser.add_argument("-v", "--verbose", action='store_true', dest="verbose", default=False,
                        help="show information for each conversions")
    parser.add_argument("--debug", action='store_true', dest="debug", default=False,
                        help="show debug information")
    group = parser.add_argument_group("main arguments")
#     group.add_argument("-l", "--list", action="store_true", dest="list", default=None,
#                        help="show the list of available formats and exit")
    group.add_argument("-o", "--output", default='reciprocal_volume.cxi', type=str,
                       help="output filename in CXI format")
    group.add_argument("-s", "--shape", default=None, type=int, nargs=3,
                       help="Size of the reciprocal volume, by default 512³")
#     group.add_argument("-D", "--dummy", type=float, default=numpy.nan,
#                        help="Set masked values to this dummy value")

    group = parser.add_argument_group("optional behaviour arguments")
#     group.add_argument("-f", "--force", dest="force", action="store_true", default=False,
#                        help="if an existing destination file cannot be" +
#                        " opened, remove it and try again (this option" +
#                        " is ignored when the -n option is also used)")
#     group.add_argument("-n", "--no-clobber", dest="no_clobber", action="store_true", default=False,
#                        help="do not overwrite an existing file (this option" +
#                        " is ignored when the -i option is also used)")
#     group.add_argument("--remove-destination", dest="remove_destination", action="store_true", default=False,
#                        help="remove each existing destination file before" +
#                        " attempting to open it (contrast with --force)")
#     group.add_argument("-u", "--update", dest="update", action="store_true", default=False,
#                        help="copy only when the SOURCE file is newer" +
#                        " than the destination file or when the" +
#                        " destination file is missing")
#     group.add_argument("-i", "--interactive", dest="interactive", action="store_true", default=False,
#                        help="prompt before overwrite (overrides a previous -n" +
#                        " option)")
    group.add_argument("--dry-run", dest="dry_run", action="store_true", default=False,
                       help="do everything except modifying the file system")
    group.add_argument("--mask", dest="mask", type=str, default=None,
				       help="Path for the mask file containing both invalid pixels and beam-stop shadow")

    group = parser.add_argument_group("Experimental setup options")
#     group.add_argument("-e", "--energy", type=float, default=None,
#                        help="Energy of the incident beam in keV")
#     group.add_argument("-w", "--wavelength", type=float, default=None,
#                        help="Wavelength of the incident beam in Å")
    group.add_argument("-d", "--distance", type=float, default=None,
                       help="Detector distance in meter")
    group.add_argument("-b", "--beam", nargs=2, type=float, default=None,
                       help="Direct beam in pixels x, y, by default, the center of the image")
    group.add_argument("-p", "--pixelsize", type=float, default=172e-6,
                       help="pixel size, by default 172µm")

    group = parser.add_argument_group("Scan setup")
#     group.add_argument("--axis", type=str, default=None,
#                        help="Goniometer angle used for scanning: 'omega', 'phi' or 'kappa'")
    group.add_argument("--rot", type=str, default="ths",
                       help="Name of the rotation motor")
    group.add_argument("--scan", type=str, default="dscan sz",
                       help="Name of the rotation motor")
    group.add_argument("--scan-len", type=str, dest="scan_len", default="1",
                       help="Pick scan which match that length (unless take all scans")

    try:
        args = parser.parse_args()

        if args.debug:
            logger.setLevel(logging.DEBUG)

        if len(args.IMAGE) == 0:
            raise argparse.ArgumentError(None, "No input file specified.")

        # the upper case IMAGE is used for the --help auto-documentation
        args.images = expand_args(args.IMAGE)
        args.images.sort()
    except argparse.ArgumentError as e:
        logger.error(e.message)
        logger.debug("Backtrace", exc_info=True)
        return EXIT_ARGUMENT_FAILURE
    else:
        return args


def parse_bliss_file(filename, title="dscan sz", rotation="ths", scan_len="1"):
    """
    scan a file and search for scans suitable for  
    
    :return: dict with angle as key and image as value
    """
    res = {}
    with h5py.File(filename, mode="r") as h5:
        for entry in h5.values():
            if entry.attrs.get("NX_class") != "NXentry":
                continue
            scan_title = entry.get("title")
            if scan_title is None:
                continue
            scan_title = as_str(scan_title[()])
            if scan_title.startswith(title):
                if scan_len and scan_title.split()[-2] != scan_len:
                    continue

                for instrument in entry.values():

                    if isinstance(instrument, h5py.Group) and instrument.attrs.get("NX_class") == "NXinstrument":
                        break
                else:
                    continue
                for detector in instrument.values():
                    if (isinstance(detector, h5py.Group) and
                        detector.attrs.get("NX_class") == "NXdetector" and
                        "type" in detector and
                        "data" in detector and
                        (detector["type"][()]).lower() == "lima"):
                            break
                else:
                    continue

                for positioners in instrument.values():
                    if (isinstance(positioners, h5py.Group) and
                        positioners.attrs.get("NX_class") == "NXcollection" and
                        rotation in positioners):

                        break
                else:
                    continue
                th = positioners[rotation][()]
                ds = detector["data"]
                print(entry.name, instrument.name, detector.name, ds)
                signal = numpy.ascontiguousarray(ds[0], dtype=numpy.float32)
                if ds.shape[0] > 1:
                    signal -= numpy.ascontiguousarray(ds[1], dtype=numpy.float32)
                res[th] = signal
    return res


class Regrid3D(OpenclProcessing):
    "Project a 2D frame to a 3D volume taking into account the curvature of the Ewald's sphere"
    kernel_files = ["regrid.cl"]

    def __init__(self, image_shape, volume_shape, center, pixel_size, distance, nb_slab=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, memory=None, profile=False):
        """
        :param image_shape: 2-tuple of int
        :param volume_shape: 3-tuple of int
        :param center: 2-tuple of float (y,x)
        :param pixel_size: float
        :param distance: float
        :param nb_slab: split the work into that many slabs. Set to None to guess
        
        """
        OpenclProcessing.__init__(self, ctx=None, devicetype=devicetype, platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, memory=memory, profile=profile)
        self.image_shape = tuple(numpy.int32(i) for i in image_shape[:2])
        self.volume_shape = tuple(numpy.int32(i) for i in volume_shape[:3])
        self.center = tuple(numpy.float32(i) for i in center[:2])
        self.pixel_size = numpy.float32(pixel_size)
        self.distance = numpy.float32(distance)
        self.center = center
        if nb_slab is None:
            self.nb_slab = self.calc_slabs()
        else:
            self.nb_slab = int(nb_slab)
        self.slab_size = int(self.volume_shape[0] / self.nb_slab)
        print(self.nslab)
        buffers = [BufferDescription("image", image_shape, numpy.float32, None),
                   BufferDescription("mask", image_shape, numpy.int8, None),
                   BufferDescription("signal", (self.nb_slab,) + self.volume_shape[1:], numpy.float32, None),
                   BufferDescription("norm", (self.nb_slab,) + self.volume_shape[1:], numpy.int32, None),
                   ]
        self.allocate_buffers(buffers, use_array=True)
        self.wg = {"normalize_signal": self.check_workgroup_size("normalize_signal"),  # largest possible WG
                   "memset_signal": self.check_workgroup_size("memset_signal"),  # largest possible WG
                   "regid_CDI_slab": query_kernel_info(self.program,  # smallest readonnably possible
                                                       kernel=self.kernels["regid_CDI_slab"],
                                                       what="PREFERRED_WORK_GROUP_SIZE_MULTIPLE")}
        print(self.wg, self.nb_slab)

    def calc_slabs(self):
        "Calculate the number of slabs needed to store data in the device's memory. The fewer, the faster"
        device_mem = sel.device.memory
        image_nbytes = numpy.prod(self.image_shape) * 4
        mask_nbytes = numpy.prod(self.image_shape) * 1
        volume_nbytes = numpy.prod(self.volume_shape) * 4 * 2
        nslab = int(ceil(volume_nbytes / (0.8 * device_mem - image_nbytes - mask_nbytes)))
        return nslab

    def send_image(self, image):
        """
        Send image to the GPU
        """
        image_d = self.buffers["image"]
        assert image.shape == image_shape
        assert image.dtype.type == numpy.float32
        image_d.set(image)
        self.profile_add(image_d.events[-1], "Copy image H --> D")

    def send_mask(self, mask):
        """
        Send mask to the GPU
        """
        mask_d = self.buffers["mask"]
        assert mask_d.shape == image_shape
        assert mask_d.dtype.type == numpy.int8
        mask_d.set(mask)
        self.profile_add(mask_d.events[-1], "Copy mask H --> D")

    def project_one_frame(self, frame,
                          rot, d_rot,
                          slab_start, slab_end,
                          oversampling_img, oversampling_rot):
        """Projection of one image onto one slab
        :param frame: numpy.ndarray 2d, floa32 image
        :param rot: angle of rotation
        :param d_rot: angular step (used for oversampling_rot)
        :param slab_start: start index of the slab
        :param slab_end: stop index of the slab
        :oversampling_img: Each pixel will be split in n x n and projected that many times
        :oversampling_rot: project multiple times each image between rot and rot+d_rot 
        :return: None
        """

        self.send_image(frame)
        ws = self.wg["regid_CDI_slab"]
        ts = int(ceil(self.image_shape[1] / wg)) * wg
        evt = prg.regid_CDI_slab(self.queue, (ts, self.image_shape[0]) , (ws, 1),
                                 image_d.data,
                                 mask_d.data,
                                 * self.image_shape,
                                 self.pixel_size,
                                 self.distance,
                                 rot, d_rot,
                                 *self.center,
                                 self.buffers["signal"].data,
                                 self.buffers["norm"],
                                 self.volume_shape[-1],
                                 slab_start,
                                 slab_end,
                                 oversampling_img,
                                 oversampling_rot)
        self.profile_add(evt, "Projection onto slab")

    def project_frames(self, frames,
                       slab_start, slab_end,
                       oversampling_img, oversampling_rot,
                       callback=lambda a: None):
        """
        Project all frames onto the slab.
        
        :param frames: dict with angle as keys and fames as values.
        :param callback: function to be called at each step (i.e. progress-bar)
        :return: the slab 
        """
        callback("memset slab")
        self.clean_slab()

        angles = numpy.array(frames.keys())
        angles.sort()
        steps = angles[1:] - angles[:-1]
        step = numpy.float32(steps.min())
        if slab_end - slab_start > self.slab_size:
            raise RuntimeError("Too many data to fit into memory")
        slab_start = numpy.int32(slab_start)
        slab_end = numpy.int32(slab_end)
        oversampling_img = numpy.int32(oversampling_img)
        oversampling_rot = numpy.int32(oversampling_rot)

        for angle, frame in frames:
            callback(f"Project angle {angle}")
            self.project_one_frame(frame,
                                   angle, step,
                                   slab_start, slab_end,
                                   oversampling_img, oversampling_rot)

        callback("get slab")
        return self.get_slab()

    def clean_slab(self):
        "Memset the slab"
        size = self.slab_size * self.volume_shape[1] * self.volume_shape[2]
        wg = self.wg["memset_signal"]
        ts = int(ceil(size / wg)) * wg
        evt = self.program.normalize_signal(self.queue, (ts,), (wg,),
                                            self.buffers["signal"].data,
                                            self.buffers["norm"].data,
                                            numpy.uint64(size))
        self.profile_add(evt, "Memset signal/count")

    def get_slab(self):
        """
        After all frames have been projected onto the slab, retrieve it after normalization 
        
        :return: Ndarray of size (slab_size, volume_size_1, volume_size_2) 
        """
        size = self.slab_size * self.volume_shape[1] * self.volume_shape[2]
        wg = self.wg["normalize_signal"]
        ts = int(ceil(size / wg)) * wg
        signal = self.buffers["signal"].data
        evt = self.program.normalize_signal(self.queue, (ts,), (wg,),
                                            signal,
                                            self.buffers["norm"].data,
                                            numpy.uint64(size))
        self.profile_add(evt, "Normalization signal/count")
        result = self.buffers["signal"].get()
        self.profile_add(signal.events[-1], "Copy slab D --> H")
        return result


def process(options, data):
    """
    Manage the process
    """
    nframes = len(data)


def main():
    """Main program
    
    :return: exit code
    """
    config = parse()
    if isinstance(config, int):
        return config
    frames = {}
    t0 = time.perf_counter()
    for fn in config.images:
        frames.update(parse_bliss_file(fn, title=config.scan, rotation=config.rot, scan_len=config.scan_len))
    print("Reading %s frames took %.3fs" % (len(frames), time.perf_counter() - t0))

    one_frame = frames[list(frames.keys())[0]]
    shape = options.shape
    if shape is None:
        shape = 512, 512, 512

    regrid = Regrid3D(one_frame.shape,
                      shape,
                      options.beam,
                      options.distance,
                      options.pixelsize,
                      profile=True)
    pb = ProgressBar("Projecting frames", nframes, 30)


def save_cxi(data, filename):
    pass


if __name__ == "__main__":
    result = main()
    sys.exit(result)
