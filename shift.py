import numpy
import fabio
import h5py
import hdf5plugin
from pyFAI.utils.mathutil import measure_offset
from matplotlib.pyplot import subplots
mdata = fabio.open("img_421.edf").data
hdata = h5py.File("regrid_mask-1-1-1.h5", "r")['/SiO2msgel3'][284]
hdata[numpy.logical_not(numpy.isfinite(hdata))] = 0
mdata = numpy.arcsinh(mdata).clip(0, 10)
hdata = numpy.arcsinh(hdata).clip(0, 10)
s = hdata.shape[0] // 2
print("Full image offset: ", measure_offset(mdata, hdata))
print("Quadrant 1 offset: ", measure_offset(mdata[:s, :s], hdata[:s, :s]))
print("Quadrant 2 offset: ", measure_offset(mdata[-s:, :s], hdata[-s:, :s]))
print("Quadrant 3 offset: ", measure_offset(mdata[:s, -s:], hdata[:s, -s:]))
print("Quadrant 4 offset: ", measure_offset(mdata[-s:, -s:], hdata[-s:, -s:]))
fig, ax = subplots(3, 1)
ax[0].imshow(mdata)
ax[0].set_title(r"Input image #415 ($\phi=0$)")
ax[1].imshow(hdata)
ax[1].set_title("Regrid volume frame #284 (center)")
ax[2].imshow(hdata - mdata)
ax[2].set_title("Difference")

fig.show()
input()
