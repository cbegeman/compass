import xarray
import numpy
import matplotlib.pyplot as plt

#path = '/lcrc/group/e3sm/ac.cbegeman/scratch/MPAS-Ocean-output/testcases_bugfixes-ramp_thin-film_chrys_2/ocean/isomip_plus/2km/single_layer/thin_film_Ocean0'
path = '/lcrc/group/e3sm/ac.cbegeman/scratch/MPAS-Ocean-output/testcases_master_thin-film_chrys/ocean/isomip_plus/5km/single_layer/Ocean0'
step = 'simulation'
ds = xarray.open_dataset(f'{path}/{step}/output.nc')
dsMesh = xarray.open_dataset(f'{path}/{step}/init.nc')
dsForcing = xarray.open_dataset(f'{path}/{step}/forcing_data_init.nc')

ssh = ds.ssh
landIcePressureInit = dsMesh.landIcePressure
layerThickness = ds.layerThickness
#landIcePressure = ds.landIcePressure
xCell = dsMesh.xCell
yCell = dsMesh.yCell
nt = min(ds.sizes['Time'],10)

tidalInputMask = dsForcing.tidalInputMask
mask = tidalInputMask.values==1
print(f'sum(mask)={numpy.sum(mask)}')
openOceanMask = landIcePressureInit.values[0,:]==0
print(f'sum(openOceanMask)={numpy.sum(openOceanMask)}')

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
    sshOpenOcean = ssh[tidx,openOceanMask]
    print(f'sshMin,sshMean,sshMax: {numpy.min(sshForcing.values)},{numpy.mean(sshForcing.values)},{numpy.max(sshForcing.values)}')
    print(f'plot: {days[tidx]},{numpy.nanmean(sshForcing.values)}')
    plt.plot(days[tidx],numpy.nanmean(sshForcing.values),'-',label='boundary')
    plt.plot(days[tidx],numpy.nanmean(sshOpenOcean.values),'--',label='open ocean')
plt.legend()
plt.savefig(f'{path}/viz/plots/ssh_t.png')
plt.close(fig)

# Plot time evolution of layer thickness in thin film region
thinFilmMask = xCell<450e3
print(f'type(layerThickness)={type(layerThickness)}')
print(f'shape(layerThickness)={numpy.shape(layerThickness)}')
print(f'shape(thinFilmMask)={numpy.shape(thinFilmMask)}')

if numpy.sum(thinFilmMask.values) > 0:
    fig = plt.figure()
    for tidx in range(nt):
        h = layerThickness[tidx,thinFilmMask,:].values
        print(f'shape(h)={numpy.shape(h)}')
        #h = h.flatten()
        #print(f'shape(h.flatten)={numpy.shape(h)}')
        #print(f'sshMin,sshMean,sshMax: {numpy.min(sshForcing.values)},{numpy.mean(sshForcing.values)},{numpy.max(sshForcing.values)}')
        #print(f'plot: {days[tidx]},{numpy.mean(h.values)}')
        plt.plot(days[tidx],numpy.mean(h),'.k')
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
