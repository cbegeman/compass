import xarray
import numpy
import matplotlib.pyplot as plt

#path = '/lcrc/group/e3sm/ac.cbegeman/scratch/MPAS-Ocean-output/testcases_bugfixes-ramp_thin-film_chrys_2/ocean/isomip_plus/2km/single_layer/thin_film_Ocean0'
path = '/lcrc/group/e3sm/ac.cbegeman/scratch/MPAS-Ocean-output/testcases_master_thin-film_chrys/ocean/isomip_plus/2km/single_layer/thin_film_Ocean0'
#ds = xarray.open_dataset(f'{path}/simulation/output.nc')
ds = xarray.open_dataset(f'{path}/performance/output.nc')
dsMesh = xarray.open_dataset(f'{path}/simulation/init.nc')
dsForcing = xarray.open_dataset(f'{path}/simulation/forcing_data_init.nc')

ssh = ds.ssh
layerThickness = ds.layerThickness
#landIcePressure = ds.landIcePressure
xCell = dsMesh.xCell
yCell = dsMesh.yCell
nt = ds.sizes['Time']

tidalInputMask = dsForcing.tidalInputMask
mask = tidalInputMask.values==1

time = ds.daysSinceStartOfSim.values
#days = time.astype('timedelta64[D]').astype(float)
days = time.astype('timedelta64[s]').astype(float)

x = xCell[mask]
y = yCell[mask]
print(f'xMin,xMean,xMax: {numpy.min(x.values)},{numpy.mean(x.values)},{numpy.max(x.values)}')
print(f'yMin,yMean,yMax: {numpy.min(y.values)},{numpy.mean(y.values)},{numpy.max(y.values)}')

fig = plt.figure()
for tidx in range(nt):
    sshForcing = ssh[tidx,mask]
    #print(f'sshMin,sshMean,sshMax: {numpy.min(sshForcing.values)},{numpy.mean(sshForcing.values)},{numpy.max(sshForcing.values)}')
    #print(f'plot: {time[tidx]},{numpy.mean(sshForcing.values)}')
    plt.plot(days[tidx],numpy.mean(sshForcing.values))
plt.savefig(f'{path}/viz/plots/ssh_t.png')
plt.close(fig)

# Plot time evolution of layer thickness in thin film region
thinFilmMask = xCell<450e3
fig = plt.figure()
for tidx in range(nt):
    h = layerThickness[tidx,thinFilmMask,:]
    #print(f'sshMin,sshMean,sshMax: {numpy.min(sshForcing.values)},{numpy.mean(sshForcing.values)},{numpy.max(sshForcing.values)}')
    #print(f'plot: {days[tidx]},{numpy.mean(h.values)}')
    plt.plot(days[tidx],numpy.mean(h.values),'.k')
plt.savefig(f'{path}/viz/plots/h_thin_film_t.png')
plt.close(fig)

fig = plt.figure()
for tidx in range(nt):
    h = ssh[tidx,thinFilmMask]
    plt.plot(days[tidx],numpy.mean(h.values),'.k')
plt.savefig(f'{path}/viz/plots/ssh_thin_film_t.png')
plt.close(fig)

#fig = plt.figure()
#for tidx in range(nt):
#    h = landIcePressure[tidx,thinFilmMask]
#    plt.plot(days[tidx],numpy.mean(p.values),'.k')
#plt.savefig(f'{path}/viz/plots/p_thin_film_t.png')
#plt.close(fig)
