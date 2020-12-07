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
__date__ = "07/12/2020"

import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from math import ceil
import numpy
import pyopencl
from pyopencl import array as cla
import time
import glob
import fabio
import h5py
import hdf5plugin
from pyFAI.utils.shell import ProgressBar
from silx.opencl.processing import OpenclProcessing, BufferDescription, KernelContainer
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
                       help="Size of the reciprocal volume (3 int), by default 512³")
#     group.add_argument("-D", "--dummy", type=float, default=numpy.nan,
#                        help="Set masked values to this dummy value")
    group.add_argument("-m", "--mask", dest="mask", type=str, default=None,
                       help="Path for the mask file containing both invalid pixels and beam-stop shadow")

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
    group = parser.add_argument_group("Oversampling options to reduces the moiré pattern")
    group.add_argument("--oversampling-img", type=int, dest="oversampling_img", default=8,
                       help="How many sub-pixel there are in one pixel (squared)")
    group.add_argument("--oversampling-rot", type=int, dest="oversampling_rot", default=8,
                       help="How many times a frame is projected")
    group = parser.add_argument_group("OpenCL options")
    group.add_argument("--device", type=int, default=None, nargs=2,
                       help="Platform and device ids")
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


def parse_bliss_file(filename, title="dscan sz", rotation="ths", scan_len="1", callback=lambda a, increment:None):
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
                callback(detector.name, increment=False)
                th = positioners[rotation][()]
                ds = detector["data"]
                signal = numpy.ascontiguousarray(ds[0], dtype=numpy.float32)
                if ds.shape[0] > 1:
                    signal -= numpy.ascontiguousarray(ds[1], dtype=numpy.float32)
                res[th] = signal
                if len (res) > 10: break
    return res


class Regrid3D(OpenclProcessing):
    "Project a 2D frame to a 3D volume taking into account the curvature of the Ewald's sphere"
    kernel_files = ["regrid.cl"]

    def __init__(self, mask, volume_shape, center, pixel_size, distance, nb_slab=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, memory=None, profile=False):
        """
        :param mask: numpy array with the mask: needs to be of the same shape as the image
        :param volume_shape: 3-tuple of int
        :param center: 2-tuple of float (y,x)
        :param pixel_size: float
        :param distance: float
        :param nb_slab: split the work into that many slabs. Set to None to guess
        
        """
        OpenclProcessing.__init__(self, ctx=None, devicetype=devicetype, platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, memory=memory, profile=profile)
        mask = numpy.ascontiguousarray(mask, dtype=numpy.int8)
        self.image_shape = tuple(numpy.int32(i) for i in mask.shape)
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
        buffers = [BufferDescription("image", image_shape, numpy.float32, None),
                   BufferDescription("mask", image_shape, numpy.int8, None),
                   BufferDescription("signal", (self.slab_size,) + self.volume_shape[1:], numpy.float32, None),
                   BufferDescription("norm", (self.slab_size,) + self.volume_shape[1:], numpy.int32, None),
                   ]
        print(buffers)
        self.allocate_buffers(buffers, use_array=True)
        self.compile_kernels([os.path.join(os.path.dirname(os.path.abspath(__file__)), "regrid.cl")])
        self.wg = {"normalize_signal": self.kernels.max_workgroup_size("normalize_signal"),  # largest possible WG
                   "memset_signal": self.kernels.max_workgroup_size("memset_signal"),  # largest possible WG
                   "regid_CDI_slab": self.kernels.min_workgroup_size("regid_CDI_slab")}
        print(self.wg, self.nb_slab)
        self.send_mask(self.mask)

    def calc_slabs(self):
        "Calculate the number of slabs needed to store data in the device's memory. The fewer, the faster"

        nslab = 1
        # Limit one slab to 2^32 in size ?
        # int(ceil((1 << 32) / numpy.prod(self.volume_shape)))

        device_mem = self.device.memory
        image_nbytes = numpy.prod(self.image_shape) * 4
        mask_nbytes = numpy.prod(self.image_shape) * 1
        volume_nbytes = numpy.prod(self.volume_shape) * 4 * 2
        nslab = max(nslab, int(ceil(volume_nbytes / (0.8 * device_mem - image_nbytes - mask_nbytes))))

        # Limit one slab to the maximum allocatable memory
        device_mem = self.ctx.devices[0].max_mem_alloc_size
        volume_nbytes = numpy.prod(self.volume_shape) * 4
        nslab = max(nslab, int(ceil(volume_nbytes / device_mem)))

        return nslab

    def compile_kernels(self, kernel_files=None, compile_options=None):
        """Call the OpenCL compiler

        :param kernel_files: list of path to the kernel
            (by default use the one declared in the class)
        :param compile_options: string of compile options
        """
        # concatenate all needed source files into a single openCL module
        kernel_files = kernel_files or self.kernel_files
        kernel_src = "\n".join(open(i).read() for i in kernel_files)

        compile_options = compile_options or self.get_compiler_options()
        logger.info("Compiling file %s with options %s", kernel_files, compile_options)
        try:
            self.program = pyopencl.Program(self.ctx, kernel_src).build(options=compile_options)
        except (pyopencl.MemoryError, pyopencl.LogicError) as error:
            raise MemoryError(error)
        else:
            self.kernels = KernelContainer(self.program)

    def send_image(self, image):
        """
        Send image to the GPU
        """
        image_d = self.cl_mem["image"]
        assert image.shape == self.image_shape
        self.profile_add(image_d.events[-1], "Copy image H --> D")

    def send_mask(self, mask):
        """
        Send mask to the GPU
        """
        mask_d = self.cl_mem["mask"]
        assert mask_d.shape == self.image_shape
        image_d.set(numpy.ascontiguousarray(mask, dtype=numpy.int8))
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
        wg = self.wg["regid_CDI_slab"]
        ts = int(ceil(self.image_shape[1] / wg)) * wg
        evt = self.program.regid_CDI_slab(self.queue, (ts, self.image_shape[0]) , (wg, 1),
                                          image_d.data,
                                          mask_d.data,
                                          * self.image_shape,
                                          self.pixel_size,
                                          self.distance,
                                          rot, d_rot,
                                          *self.center,
                                          self.cl_mem["signal"].data,
                                          self.cl_mem["norm"].data,
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

        angles = numpy.array(list(frames.keys()))
        angles.sort()
        steps = angles[1:] - angles[:-1]
        step = numpy.float32(steps.min())
        if slab_end - slab_start > self.slab_size:
            raise RuntimeError("Too many data to fit into memory")
        slab_start = numpy.int32(slab_start)
        slab_end = numpy.int32(slab_end)
        oversampling_img = numpy.int32(oversampling_img)
        oversampling_rot = numpy.int32(oversampling_rot)

        for angle, frame in frames.items():
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
        print(wg, ts, size, self.slab_size, self.volume_shape, self.cl_mem["signal"].shape, self.cl_mem["norm"].shape)
        evt = self.program.memset_signal(self.queue, (ts,), (wg,),
                                         self.cl_mem["signal"].data,
                                         self.cl_mem["norm"].data,
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
        signal = self.cl_mem["signal"].data
        evt = self.program.normalize_signal(self.queue, (ts,), (wg,),
                                            signal,
                                            self.cl_mem["norm"].data,
                                            numpy.uint64(size))
        self.profile_add(evt, "Normalization signal/count")
        result = self.cl_mem["signal"].get()
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
    print("Regrid diffraction images in 3D reciprocal space")

    mask = fabio.open(config.mask).data
    shape = config.shape
    if shape is None:
        shape = 512, 512, 512

    if config.device is None:
        pid, did = None, None
    else:
        pid, did = config.device

    regrid = Regrid3D(mask,
                      shape,
                      config.beam,
                      config.distance,
                      config.pixelsize,
                      profile=True,
                      platformid=pid,
                      deviceid=did)

    print("Working on device", regrid.device.name)
    pb = ProgressBar("Reading frames", 100, 30)

    def callback(msg, increment=True, cnt={"value": 0}):
        if increment:
            cnt["value"] += 1
        pb.update(cnt["value"], msg)

    t0 = time.perf_counter()
    for fn in config.images:
        frames.update(parse_bliss_file(fn, title=config.scan, rotation=config.rot, scan_len=config.scan_len, callback=callback))
    t1 = time.perf_counter()

    pb.max_value = (len(frames) + 2) * regrid.nb_slab
    slab_heigth = config.shape[0] // regrid.nb_slab
    for slab_start in numpy.arange(0, config.shape[0], slab_heigth, dtype=numpy.int32):
        slab_end = min(slab_start + slab_heigth, config.shape[0])
        pb.title = "Projection onto slab %i-%i" % (slab_start, slab_end)
        slab = regrid.project_frames(frames,
                                     slab_start, slab_end,
                                     config.oversampling_img,
                                     config.oversampling_rot,
                                     callback)
        print(slab.shape)
    t2 = time.perf_counter()

    print(t1 - t0, t2 - t1, len(frames))


def save_cxi(data, filename):
    pass


if __name__ == "__main__":
    result = main()
    sys.exit(result)
