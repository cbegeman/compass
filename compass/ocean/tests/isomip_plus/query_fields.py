import xarray
import numpy
import matplotlib.pyplot as plt

path = '/lcrc/group/e3sm/ac.cbegeman/scratch/MPAS-Ocean-output/testcases_bugfixes-ramp_thin-film_chrys_2/ocean/isomip_plus/2km/single_layer/thin_film_Ocean0'
ds = xarray.open_dataset(f'{path}/performance/output.nc')
#dsMesh = xarray.open_dataset(f'{path}/simulation/init.nc')
#dsForcing = xarray.open_dataset(f'{path}/simulation/forcing_data_init.nc')

layerThickness = ds.layerThickness
#xCell = dsMesh.xCell
#yCell = dsMesh.yCell
nt = ds.sizes['Time']

h = layerThickness[nt-1,:]
#tidalInputMask = dsForcing.tidalInputMask
#mask = tidalInputMask.values==1

print(f'hMin,hMean,hMax: {numpy.min(h.values)},{numpy.mean(h.values)},{numpy.max(h.values)}')
