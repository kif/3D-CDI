#!/usr/bin/env python3

"""
Rebuild the 3D reciprocal space  
by projecting a set of 2d speckle SAXS pattern taken at various rotation angles 
into a 3D regular volume
"""

__author__ = "Jérôme Kieffer"
__copyright__ = "2020 ESRF"
__license__ = "MIT"
__version__ = "0.1"

import os
import sys
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


def main():

    epilog = """Assumption: There is enough memory to hold all frames in memory
     
                return codes: 0 means a success. 1 means the conversion
                contains a failure, 2 means there was an error in the
                arguments"""

    parser = argparse.ArgumentParser(prog="cdi-regrid",
                                     description=__doc__,
                                     epilog=epilog)
    parser.add_argument("IMAGE", nargs="*",
                        help="File with input images")
    parser.add_argument("-V", "--version", action='version', version=__version__,
                        help="output version and exit")
    parser.add_argument("-v", "--verbose", action='store_true', dest="verbose", default=False,
                        help="show information for each conversions")
    parser.add_argument("--debug", action='store_true', dest="debug", default=False,
                        help="show debug information")
    group = parser.add_argument_group("main arguments")
#     group.add_argument("-l", "--list", action="store_true", dest="list", default=None,
#                        help="show the list of available formats and exit")
    group.add_argument("-o", "--output", default='reciprocal_volume.cxi', type=str,
                       help="output  filename")
    group.add_argument("-s", "--shape", default=512, type=int,
                       help="Size of the reciprocal volume, by default 512³")
#     group.add_argument("-D", "--dummy", type=int, default=-1,
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

    group = parser.add_argument_group("Experimental setup options")
    group.add_argument("-d", "--distance", type=float, default=1.0,
                       help="Detector distance in meters, by default 1m")
    group.add_argument("-b", "--beam", nargs=2, type=float, default=None,
                       help="Direct beam in pixels x, y, by default, the center of the image")
    group.add_argument("-p", "--pixel", type=float, default=172e-6,
                       help="pixel size, by default 172µm")

    group = parser.add_argument_group("Goniometer setup")
    group.add_argument("--phi", type=str, default=None,
                       help="Goniometer angle phi value in deg. from the fomula like '-80+0.2*index`")

    try:
        args = parser.parse_args()

        if args.debug:
            logger.setLevel(logging.DEBUG)

#         if args.list:
#             print_supported_formats()
#             return

        if len(args.IMAGE) == 0:
            raise argparse.ArgumentError(None, "No input file specified.")

        # the upper case IMAGE is used for the --help auto-documentation
        args.images = expand_args(args.IMAGE)
        args.images.sort()
    except argparse.ArgumentError as e:
        logger.error(e.message)
        logger.debug("Backtrace", exc_info=True)
        return EXIT_ARGUMENT_FAILURE

    succeeded = convert_all(args)
    if not succeeded:
        print("Conversion or part of it failed. You can try with --debug to have more output information.")
        return EXIT_FAILURE

    return EXIT_SUCCESS


def parse_files():
    "Return the number of frames, and the shape of the input dataset"
    pass


def save_cxi(data, filename):
    pass


if __name__ == "__main__":
    result = main()
    sys.exit(result)
