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
__date__ = "02/12/2020"

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
    group.add_argument("-s", "--shape", default=512, type=int,
                       help="Size of the reciprocal volume, by default 512³")
    group.add_argument("-D", "--dummy", type=float, default=numpy.nan,
                       help="Set masked values to this dummy value")

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
    group.add_argument("-e", "--energy", type=float, default=None,
                       help="Energy of the incident beam in keV")
    group.add_argument("-w", "--wavelength", type=float, default=None,
                       help="Wavelength of the incident beam in Å")
    group.add_argument("-d", "--distance", type=float, default=None,
                       help="Detector distance in meter")
    group.add_argument("-b", "--beam", nargs=2, type=float, default=None,
                       help="Direct beam in pixels x, y, by default, the center of the image")
    group.add_argument("-p", "--pixel", type=float, default=172e-6,
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


def main():
    """Main program
    
    :return: exit code
    """
    config = parse()
    if isinstance(config, int):
        return config
    frames = {}
    for fn in config.images:
        frames.update(parse_bliss_file(fn, title=config.scan, rotation=config.rot, scan_len=config.scan_len))
    print(len(frames))


def save_cxi(data, filename):
    pass


if __name__ == "__main__":
    result = main()
    sys.exit(result)
