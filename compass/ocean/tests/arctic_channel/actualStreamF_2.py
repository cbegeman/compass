import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy
import xarray
from netCDF4 import Dataset
from numpy import array

ds = xarray.open_dataset('output.nc')
figsize = [6.4, 4.8]
ds1 = ds.isel(Time=12)
ds1 = ds1.sortby('yEdge')
#ds1 = ds1.groupby('yEdge')

nCells = ds1.sizes['nCells']
nEdges = ds1.sizes['nEdges']
nVertLevels = ds1.sizes['nVertLevels']

xEdge = numpy.zeros((nEdges))
xEdge = ds1.xEdge
yEdge = ds1.yEdge
yCell = numpy.zeros((nCells))
yCell = ds1.yCell
xCell = ds1.xCell

xEdge_mid = numpy.median(xEdge)
xCell_mid = 151000.  # numpy.median(xCell)
#print(len(numpy.unique(xEdge)))
edgeMask_x = numpy.equal(xEdge, xEdge_mid)
dx = 1.
cellMask_x = numpy.logical_and(xCell > xCell_mid - dx,
                               xCell < xCell_mid + dx)

zIndex = xarray.DataArray(data=numpy.arange(nVertLevels),
                                      dims='nVertLevels')
zInterface = numpy.zeros((nCells, nVertLevels + 1))
zInterface[:, 0] = ds1.ssh.values
for zIndex in range(nVertLevels):
    thickness = ds1.layerThickness.isel(nVertLevels=zIndex)
    thickness = thickness.fillna(0.)
    zInterface[:, zIndex + 1] = \
        zInterface[:, zIndex] - thickness.values

zMid = numpy.zeros((nCells, nVertLevels))
for zIndex in range(nVertLevels):
    zMid[:, zIndex] = (zInterface[:, zIndex] +
                       numpy.divide(zInterface[:, zIndex + 1] -
                                    zInterface[:, zIndex], 2.))

cellsOnEdge = ds1.cellsOnEdge
cellsOnEdge_x = cellsOnEdge[edgeMask_x, :]
yEdges = numpy.zeros((len(cellsOnEdge_x) + 1))
for i in range(len(cellsOnEdge_x)):
    if cellsOnEdge[i, 1] == 0:
        yEdges[i] = yCell[cellsOnEdge_x[i, 0] - 1]
        yEdges[i + 1] = yCell[cellsOnEdge_x[i, 0] - 1]
    elif cellsOnEdge[i, 1] == 0:
        yEdges[i] = yCell[cellsOnEdge_x[i, 1] - 1]
        yEdges[i + 1] = yCell[cellsOnEdge_x[i, 1] - 1]
    else:
        yEdges[i] = min(yCell[cellsOnEdge_x[i, 0] - 1],
                        yCell[cellsOnEdge_x[i, 1] - 1])
        yEdges[i + 1] = max(yCell[cellsOnEdge_x[i, 0] - 1],
                            yCell[cellsOnEdge_x[i, 1] - 1])

#print(ds1.xCell[cellsOnEdge_x - 1].values)
#print(yEdges[1:]-yEdges[:-1])
zInterfaces_mesh, yEdges_mesh = numpy.meshgrid(zInterface[0, :], yEdges)

#zMid_mesh, yCell_mesh = numpy.meshgrid(zMid_xMid, yCell_xMid)
zMid_xMid, yEdge_xMid = numpy.meshgrid(zMid[0, :], yEdge[edgeMask_x])
zMidCell_xMid, yCell_xMid = numpy.meshgrid(zMid[0, :], yCell[cellMask_x])
#print(numpy.shape(zMid_xMid))
#print(numpy.shape(streamFuncSubmeso_xmesh.values))

density = ds1.density.values
print(f'nCells = {nCells}')
print(f'zMid_xMid shape = {numpy.shape(zMid[0, :])}')
print(f'yCell_xMid shape = {numpy.shape(yCell[cellMask_x])}')
print(f'zMidCell_xMid shape = {numpy.shape(zMidCell_xMid)}')
print(f'density_xMid shape = {numpy.shape(density[cellMask_x, :])}')

# Figures
if 'streamFuncSubmeso' in ds1.keys():
    streamFuncSubmeso = numpy.zeros((nCells, nVertLevels))
    streamFuncSubmeso = ds1.streamFuncSubmeso
    streamFuncSubmeso_xmesh = streamFuncSubmeso[edgeMask_x, :]

    plt.figure(figsize=figsize, dpi=100)
    cmax = numpy.max(numpy.abs(streamFuncSubmeso_xmesh.values))
    plt.pcolormesh(numpy.divide(yEdges_mesh, 1e3),
                   zInterfaces_mesh,
                   streamFuncSubmeso_xmesh.values,
                   cmap='RdBu', vmin=-1. * cmax, vmax=cmax)
    plt.xlabel('y (km)')
    plt.ylabel('z (m)')
    cbar = plt.colorbar()
    #cbar.ax.set_title('uNormal (m/s)')
    #plt.savefig('uNormal_depth_section_t{}.png'.format(j),
    #            bbox_inches='tight', dpi=200)
    plt.savefig('streamFunc_section.png')
    plt.close()

    plt.figure(figsize=figsize, dpi=100)
    cmax = numpy.max(numpy.abs(streamFuncSubmeso_xmesh.values))
    plt.scatter(numpy.divide(yEdge_xMid, 1e3),
                zMid_xMid,
                c=streamFuncSubmeso_xmesh.values,
                cmap='RdBu', vmin=-1. * cmax, vmax=cmax)
    plt.xlabel('y (km)')
    plt.ylabel('z (m)')
    cbar = plt.colorbar()
    #cbar.ax.set_title('uNormal (m/s)')
    #plt.savefig('uNormal_depth_section_t{}.png'.format(j),
    #            bbox_inches='tight', dpi=200)
    plt.savefig('streamFunc_section_scatter.png')
    plt.close()

plt.figure(figsize=figsize, dpi=100)
cmax = numpy.max(numpy.abs(ds1.density.values))
plt.scatter(numpy.divide(yCell_xMid, 1e3),
            zMidCell_xMid,
            c=density[cellMask_x, :])
plt.xlabel('y (km)')
plt.ylabel('z (m)')
cbar = plt.colorbar()
#cbar.ax.set_title('uNormal (m/s)')
#plt.savefig('uNormal_depth_section_t{}.png'.format(j),
#            bbox_inches='tight', dpi=200)
plt.savefig('density_section_scatter.png')
plt.close()
