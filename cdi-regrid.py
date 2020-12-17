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
__version__ = "0.9"
__date__ = "17/12/2020"

import os
import sys
import logging
logging.basicConfig(level=logging.INFO)
from math import ceil, floor
import numpy
import pyopencl
from pyopencl import array as cla
import time
import glob
import fabio
import h5py
import hdf5plugin
from silx.opencl.processing import OpenclProcessing, BufferDescription, KernelContainer
from silx.opencl.common import query_kernel_info
from pynx.cdi.cdi import save_cdi_data_cxi
import codecs
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


class ProgressBar:
    """
    Progress bar in shell mode
    """

    def __init__(self, title, max_value, bar_width):
        """
        Create a progress bar using a title, a maximum value and a graphical size.

        The display is done with stdout using carriage return to to hide the
        previous progress. It is not possible to use stdout for something else
        whill a progress bar is in use.

        The result looks like:

        .. code-block:: none

            Title [■■■■■■      ]  50%  Message

        :param str title: Title displayed before the progress bar
        :param float max_value: The maximum value of the progress bar
        :param int bar_width: Size of the progressbar in the screen
        """
        self.title = title
        self.max_value = max_value
        self.bar_width = bar_width
        self.last_size = 0
        self._message = ""
        self._value = 0.0

        encoding = None
        if hasattr(sys.stdout, "encoding"):
            # sys.stdout.encoding can't be used in unittest context with some
            # configurations of TestRunner. It does not exists in Python2
            # StringIO and is None in Python3 StringIO.
            encoding = sys.stdout.encoding
        if encoding is None:
            # We uses the safer aproch: a valid ASCII character.
            self.progress_char = '#'
        else:
            try:
                import datetime
                if str(datetime.datetime.now())[5:10] == "02-14":
                    self.progress_char = u'\u2665'
                else:
                    self.progress_char = u'\u25A0'
                _byte = codecs.encode(self.progress_char, encoding)
            except (ValueError, TypeError, LookupError):
                # In case the char is not supported by the encoding,
                # or if the encoding does not exists
                self.progress_char = '#'

    def clear(self):
        """
        Remove the progress bar from the display and move the cursor
        at the beginning of the line using carriage return.
        """
        sys.stdout.write('\r' + " " * self.last_size + "\r")
        sys.stdout.flush()

    def display(self):
        """
        Display the progress bar to stdout
        """
        self.update(self._value, self._message)

    def update(self, value=None, message="", max_value=None):
        """
        Update the progrss bar with the progress bar's current value.

        Set the progress bar's current value, compute the percentage
        of progress and update the screen with. Carriage return is used
        first and then the content of the progress bar. The cursor is
        at the begining of the line.

        :param float value: progress bar's current value
        :param str message: message displayed after the progress bar
        :param float max_value: If not none, update the maximum value of the
            progress bar
        """
        if max_value is not None:
            self.max_value = max_value
        self._message = message
        if value is None:
            value = self._value + 1
        self._value = value

        if self.max_value == 0:
            coef = 1.0
        else:
            coef = (1.0 * value) / self.max_value
        percent = round(coef * 100)
        bar_position = int(coef * self.bar_width)
        if bar_position > self.bar_width:
            bar_position = self.bar_width

        # line to display
        line = '\r%15s [%s%s] % 3d%%  %s' % (self.title, self.progress_char * bar_position, ' ' * (self.bar_width - bar_position), percent, message)

        # trailing to mask the previous message
        line_size = len(line)
        clean_size = self.last_size - line_size
        if clean_size < 0:
            clean_size = 0
        self.last_size = line_size

        sys.stdout.write(line + " " * clean_size + "\r")
        sys.stdout.flush()


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
    group.add_argument("-s", "--shape", default=1024, type=int,
                       help="Size of the reciprocal volume, by default 512³")
    group.add_argument("--scale", default=1.0, type=float,
                       help="Scale (down) the voxel coordinates. For example a factor 2 is similar to a 2x2x2 binning of the volume")

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
    group.add_argument("--profile", action="store_true", default=False,
                       help="Turn on the profiler and print OpenCL profiling at output")
    group.add_argument("--maxi", default=None, type=int,
                       help="Limit the processing to a given number of frames")

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


def parse_bliss_file(filename, title="dscan sz", rotation="ths", scan_len="1", callback=lambda a, increment:None, maxi=None):
    """Scan a Bliss file and search for scans suitable for CXI image reconstruction
    
    :param filname: str, name of the Bliss-Nexus file
    :param title: the kind of scan one is looking for
    :param rotation: name of the motor responsible for the rotation of the sample
    :param scan_len: search only for scan of this length
    :param callback: used for the progress-bar update
    :param maxi: limit the search to this number of frames (used to speed-up reading in debug mode)
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

                    if (isinstance(instrument, h5py.Group) and
                        as_str(instrument.attrs.get("NX_class")) == "NXinstrument"):
                        break
                else:
                    continue
                for detector in instrument.values():
                    if (isinstance(detector, h5py.Group) and
                        as_str(detector.attrs.get("NX_class", "")) == "NXdetector" and
                        "type" in detector and
                        "data" in detector and
                        (as_str(detector["type"][()])).lower() == "lima"):
                            break
                else:
                    continue

                for positioners in instrument.values():
                    if (isinstance(positioners, h5py.Group) and
                        as_str(positioners.attrs.get("NX_class")) == "NXcollection" and
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
                if maxi and len(res) > maxi:
                    break

    return res


class Regrid3D(OpenclProcessing):
    "Project a 2D frame to a 3D volume taking into account the curvature of the Ewald's sphere"
    kernel_files = ["regrid.cl"]

    def __init__(self, mask, volume_shape, center, pixel_size, distance, slab_size=None, scale=None,
                 ctx=None, devicetype="all", platformid=None, deviceid=None,
                 block_size=None, memory=None, profile=False):
        """
        :param mask: numpy array with the mask: needs to be of the same shape as the image
        :param volume_shape: 3-tuple of int
        :param center: 2-tuple of float (y,x)
        :param pixel_size: float, size of the pixel in meter
        :param distance: float, sample detector distance in meter
        :param slab_size: Number of slices to be treated at one, the best is to leave the system guess
        :param scale: zoom factor, 2 is like a 2x2x2 binning of the volume. Allows to work with smaller volumes.
        :param ctx: actual working context, left to None for automatic
                    initialization from device type or platformid/deviceid
        :param devicetype: type of device, can be "CPU", "GPU", "ACC" or "ALL"
        :param platformid: integer with the platform_identifier, as given by clinfo
        :param deviceid: Integer with the device identifier, as given by clinfo
        :param block_size: preferred workgroup size, may vary depending on the
                            out come of the compilation
        :param memory: minimum memory available on device
        :param profile: switch on profiling to be able to profile at the kernel
                         level, store profiling elements (makes code slightly slower)
        """
        OpenclProcessing.__init__(self, ctx=None, devicetype=devicetype, platformid=platformid, deviceid=deviceid,
                                  block_size=block_size, memory=memory, profile=profile)

        self.image_shape = tuple(numpy.int32(i) for i in mask.shape)
        logger.info("image_shape: %s", self.image_shape)
        self.volume_shape = tuple(numpy.int32(i) for i in volume_shape[:3])
        logger.info("volume_shape: %s", self.volume_shape)
        self.center = tuple(numpy.float32(i) for i in center)
        logger.info("center y,x: %s", self.center)
        self.pixel_size = numpy.float32(pixel_size)
        logger.info("pixel_size: %s", self.pixel_size)
        self.distance = numpy.float32(distance)
        logger.info("distance: %s", self.distance)
        self.scale = numpy.float32(1 if scale is None else scale)
        logger.info("scale: %s", self.scale)
        if slab_size:
            self.slab_size = int(slab_size)
        else:
            self.slab_size = self.calc_slabs()

        self.nb_slab = int(ceil(self.volume_shape[1] / self.slab_size))
        # Homogenize the slab size ... no need to stress the GPU memory
        self.slab_size = int(ceil(self.volume_shape[1] / self.nb_slab))

        self.slicing = self.calc_slicing()
        for k, v in self.slicing.items():
            print(f"slab {k}: image {v}")

        buffers = [BufferDescription("image", self.image_shape, numpy.float32, None),
                   BufferDescription("mask", self.image_shape, numpy.uint8, None),
                   BufferDescription("signal", (self.volume_shape[0], self.slab_size, self.volume_shape[2]), numpy.float32, None),
                   BufferDescription("norm", (self.volume_shape[0], self.slab_size, self.volume_shape[2]), numpy.int32, None),
                   ]
        self.allocate_buffers(buffers, use_array=True)
        self.compile_kernels([os.path.join(os.path.dirname(os.path.abspath(__file__)), "regrid.cl")])
        self.wg = {"normalize_signal": self.kernels.max_workgroup_size("normalize_signal"),  # largest possible WG
                   "memset_signal": self.kernels.max_workgroup_size("memset_signal"),  # largest possible WG
                   "regid_CDI_slab": self.kernels.min_workgroup_size("regid_CDI_slab")}
        self.send_mask(mask)
        self.progress_bar = None

    def calc_slabs(self):
        "Calculate the height of the slab depending on the device's memory. The larger, the better"
        float_size = numpy.dtype(numpy.float32).itemsize
        int_size = numpy.dtype(numpy.int32).itemsize
        device_mem = self.device.memory
        image_nbytes = numpy.prod(self.image_shape) * float_size
        mask_nbytes = numpy.prod(self.image_shape) * 1
        volume_nbytes = self.volume_shape[0] * self.volume_shape[2] * (float_size + int_size)
        tm_slab = (0.8 * device_mem - image_nbytes - mask_nbytes) / volume_nbytes

        device_mem = self.ctx.devices[0].max_mem_alloc_size
        volume_nbytes = self.volume_shape[0] * self.volume_shape[2] * float_size
        am_slab = device_mem / volume_nbytes
        logger.info("calc_slabs %s total mem says: %s allocatable mem says: %s", self.volume_shape[1], tm_slab, am_slab)
        return  int(min(self.volume_shape[0], tm_slab, am_slab))

    def calc_slicing(self):
        "Calculate the slicing, i.e, for which slab in output, which lines of the image are needed"
        shape = self.volume_shape[1]  # Number of lines in y
        shape_2 = shape // 2
        size = self.slab_size  # , slicing along y
        dist = self.distance
        center = self.center[0]  # Along y
        d0 = max(self.image_shape[0] - self.center[0], self.center[0])
        d1 = max(self.image_shape[1] - self.center[1], self.center[1])
        scale = numpy.sqrt(dist ** 2 + self.pixel_size * (d0 ** 2 + d1 ** 2)) / dist  # >1
        res = {}
        for slab_start in range(0, shape, size):
            slab = (slab_start, min(shape, slab_start + size))
            lower = min((slab[0] - shape_2) / self.scale + center,
                        (slab[0] - shape_2) * scale / self.scale + center)
            upper = max((slab[1] - shape_2) / self.scale + center,
                        (slab[1] - shape_2) * scale / self.scale + center)
            res[slab] = (max(0, int(floor(lower))),
                         min(self.image_shape[0], int(ceil(upper))))
        return res

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

    def send_image(self, image, slice_=None):
        """
        Send image to the GPU
        
        :param image: 2d numpy array
        :param slice: slice_ object with the start and end of the buffer to be copied
        :return: Nothing
        """
        image_d = self.cl_mem["image"]
        if slice_ is not None:
             slice_ = slice(max(0, slice_.start), min(self.image_shape[0], slice_.stop))
             img = numpy.ascontiguousarray(image, dtype=numpy.float32)
             evt = pyopencl.enqueue_copy(self.queue, image_d.data, img)
             self.profile_add(evt, "Copy image H --> D")
        else:
            assert image.shape == self.image_shape
            image_d.set(numpy.ascontiguousarray(image, dtype=numpy.float32))
            self.profile_add(image_d.events[-1], "Copy image H --> D")

    def send_mask(self, mask):
        """
        Send mask to the GPU
        """
        mask_d = self.cl_mem["mask"]
        assert mask_d.shape == self.image_shape
        mask_d.set(numpy.ascontiguousarray(mask, dtype=numpy.uint8))
        self.profile_add(mask_d.events[-1], "Copy mask H --> D")

    def project_one_frame(self, frame,
                          rot, d_rot,
                          vol_slice, img_slice=None,
                          oversampling_img=1, oversampling_rot=1):
        """Projection of one image onto one slab
        :param frame: numpy.ndarray 2d, floa32 image
        :param rot: angle of rotation
        :param d_rot: angular step (used for oversampling_rot)
        :param vol_slice: Start/end row in the volume (slab along y)
        :param img_slice: Start/end row in the image
        :oversampling_img: Each pixel will be split in n x n and projected that many times
        :oversampling_rot: project multiple times each image between rot and rot+d_rot 
        :return: None
        """

        self.send_image(frame, img_slice)
        wg = self.wg["regid_CDI_slab"]
        ts = int(ceil(self.image_shape[1] / wg)) * wg
        evt = self.program.regid_CDI_slaby(self.queue, (ts, img_slice.stop-img_slice.start) , (wg, 1),
                                           self.cl_mem["image"].data,
                                           self.cl_mem["mask"].data,
                                           *self.image_shape,
                                           img_slice.start, img_slice.stop,
                                           self.pixel_size,
                                           self.distance,
                                           rot, d_rot,
                                           *self.center,
                                           self.scale,
                                           self.cl_mem["signal"].data,
                                           self.cl_mem["norm"].data,
                                           self.volume_shape[-1],
                                           vol_slice.start,
                                           vol_slice.stop,
                                           oversampling_img,
                                           oversampling_rot)
        self.profile_add(evt, "Projection onto slab")

    def project_frames(self, l_frames, angles, step,
                       vol_slice, img_slice=None,
                       oversampling_img=1, oversampling_rot=1,
                       ):
        """
        Project all frames onto the slab.
        
        :param l_frames: list of frames
        :param l_angles: angles associated with the frame
        :param step: step size 
        :param vol_slice:  fraction of the volume to use (slicing along Y!)
        :param img_slice:  fraction of the image to use
        :return: the slab 
        """
        if self.progress_bar:
            self.progress_bar.update(message="memset slab")
        self.clean_slab()

        if vol_slice.stop - vol_slice.start > self.slab_size:
            raise RuntimeError("Too many data to fit into memory")
        for angle, frame in zip(angles, l_frames):
            if self.progress_bar:
                self.progress_bar.update(message=f"Project angle {angle:.1f}")
            self.project_one_frame(frame, angle, step,
                                   vol_slice, img_slice,
                                   oversampling_img, oversampling_rot)

        if self.progress_bar:
            self.progress_bar.update(message="get slab")
        return self.get_slab()

    def process_all(self, frames,
                    oversampling_img=1,
                    oversampling_rot=1,
                    ):
        """Project all frames and rebuild the 3D volume
        :param frames: dict with angle: frame as numpy.array
        :param oversample_img
        
        :return: 3D volume as numpy array
        """
        angles = list(frames.keys())
        angles.sort()
        nangles = numpy.array(angles, dtype=numpy.float32)
        steps = nangles[1:] - nangles[:-1]
        step = steps.min()
        l_frames = [frames[a] for a in angles]
        oversampling_img = numpy.int32(oversampling_img)
        oversampling_rot = numpy.int32(oversampling_rot)

        if self.progress_bar:
            self.progress_bar.max_value = (len(frames) + 2) * len(self.slicing)

        full_volume = numpy.empty(self.volume_shape, dtype=numpy.float32)

        for slab_idx, img_idx in self.slicing.items():
            vol_slice = slice(numpy.int32(slab_idx[0]), numpy.int32(slab_idx[1]))
            img_slice = slice(numpy.int32(img_idx[0]), numpy.int32(img_idx[1]))
            if self.progress_bar:
                self.progress_bar.title = "Projection onto slab %04i-%04i" % (vol_slice.start, vol_slice.stop)
            slab = self.project_frames(l_frames, nangles, step,
                                       vol_slice, img_slice,
                                       oversampling_img, oversampling_rot)
            full_volume[:, vol_slice,: ] = slab[:,: vol_slice.stop - vol_slice.start,:]
        return full_volume

    def clean_slab(self):
        "Memset the slab"
        size = self.slab_size * self.volume_shape[1] * self.volume_shape[2]
        wg = self.wg["memset_signal"]
        ts = int(ceil(size / wg)) * wg
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
        signal_d = self.cl_mem["signal"]
        norm_d = self.cl_mem["norm"]
        evt = self.program.normalize_signal(self.queue, (ts,), (wg,),
                                            signal_d.data,
                                            norm_d.data,
                                            numpy.uint64(size))
        self.profile_add(evt, "Normalization signal/count")
        result = signal_d.get()
        if signal_d.events:
            self.profile_add(signal_d.events[-1], "Copy slab D --> H")
        return result


def main():
    """Main program
    
    :return: exit code
    """
    config = parse()
    if isinstance(config, int):
        return config

    if len(config.images) == 0:
        raise RuntimeError("No input file provided !")

    frames = {}
    print("Regrid diffraction images in 3D reciprocal space")

    mask = fabio.open(config.mask).data
    shape = config.shape
    if shape is None:
        shape = 512, 512, 512
    else:
        shape = (shape, shape, shape)

    if config.device is None:
        pid, did = None, None
    else:
        pid, did = config.device

    full_volume = numpy.zeros(shape, dtype=numpy.float32)

    regrid = Regrid3D(mask,
                      shape,
                      config.beam[-1::-1],
                      config.pixelsize,
                      config.distance,
                      profile=config.profile,
                      platformid=pid,
                      deviceid=did)

    pb = ProgressBar("Reading frames", 100, 30)
    regrid.progress_bar = pb

    def callback(msg, increment=True, cnt={"value": 0}):
        if increment:
            cnt["value"] += 1
        pb.update(cnt["value"], msg)

    t0 = time.perf_counter()
    for fn in config.images:
        frames.update(parse_bliss_file(fn, title=config.scan, rotation=config.rot, scan_len=config.scan_len, callback=callback, maxi=config.maxi))
    if len(frames) == 0:
        raise RuntimeError("No valid images found in input file ! Check parameters `--rot`, `--scan` and `--scan-len`")
    t1 = time.perf_counter()
    full_volume = regrid.process_all(frames,
                                     oversampling_img=config.oversampling_img,
                                     oversampling_rot=config.oversampling_rot)
    t2 = time.perf_counter()
    if not config.dry_run:
        pb.title = "Save volume"
        pb.update(message=config.output)
        save_cxi(full_volume, config, mask=mask)
    t3 = time.perf_counter()
    if config.profile:
        print(os.linesep.join(regrid.log_profile()))
        print("#"*50)
        print(f"Frame reading: {t1 - t0:6.3f}s for {len(frames)} frames")
        print(f"Projection time: {t2 - t1:6.3f}s using {regrid.nb_slab} slabs")
        print(f"Save time: {t3 - t2:6.3f}s")
    if config.dry_run:
        print("Done --> None")
    else:
        print("Done -->", config.output)


def save_cxi(data, config, mask=None):
    save_cdi_data_cxi(config.output, data,
                      wavelength=None,
                      detector_distance=config.distance,
                      pixel_size_detector=config.pixelsize,
                      mask=mask,
                      sample_name=None,
                      experiment_id=None,
                      instrument=None,
                      note=None,
                      iobs_is_fft_shifted=False,
                      process_parameters=None)


if __name__ == "__main__":
    result = main()
    sys.exit(result)
