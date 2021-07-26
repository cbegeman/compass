import xarray
import numpy

from compass.ocean.haney import compute_haney_number
from compass.ocean.vertical.grid_1d import add_1d_grid
from compass.ocean.vertical.partial_cells import alter_bottom_depth, alter_ssh
from compass.ocean.vertical.zlevel import compute_z_level_layer_thickness, \
    compute_min_max_level_cell

def init_hybrid_vertical_coord(config, ds):
    """
    Create a hybrid z-star vertical coordinate based on the config options in the
    ``vertical_grid`` section and the ``bottomDepth`` and ``ssh`` variables of
    the mesh data set.

    The following new variables will be added to the data set:

      * ``minLevelCell`` - the index of the top valid layer

      * ``maxLevelCell`` - the index of the bottom valid layer

      * ``cellMask`` - a mask of where cells are valid

      * ``layerThickness`` - the thickness of each layer

      * ``restingThickness`` - the thickness of each layer stretched as if
        ``ssh = 0``

      * ``zMid`` - the elevation of the midpoint of each layer

    So far, all supported coordinates make use of a 1D reference vertical grid.
    The following variables associated with that field are also added to the
    mesh:

      * ``refTopDepth`` - the positive-down depth of the top of each ref. level

      * ``refZMid`` - the positive-down depth of the middle of each ref. level

      * ``refBottomDepth`` - the positive-down depth of the bottom of each ref.
        level

      * ``refInterfaces`` - the positive-down depth of the interfaces between
        ref. levels (with ``nVertLevels`` + 1 elements).

      * ``vertCoordMovementWeights`` - the weights (all ones) for coordinate
        movement

    There is considerable redundancy between these variables but each is
    sometimes convenient.

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    ds : xarray.Dataset
        A data set containing ``bottomDepth`` and ``ssh`` variables used to
        construct the vertical coordinate
    """
    # --- Begin with initialization of z-star coordinate ---
    add_1d_grid(config, ds)

    ds['vertCoordMovementWeights'] = xarray.ones_like(ds.refBottomDepth)

    restingSSH = xarray.zeros_like(ds.bottomDepth)
    
    ds['minLevelCell'], ds['maxLevelCell'], ds['cellMask'] = \
        compute_min_max_level_cell(ds.refTopDepth, ds.refBottomDepth, ds.ssh,
                                   ds.bottomDepth)

    ds['bottomDepth'], ds['maxLevelCell'] = alter_bottom_depth(
        config, ds.bottomDepth, ds.refBottomDepth, ds.maxLevelCell)

    ds['ssh'], ds['minLevelCell'] = alter_ssh(
        config, ds.ssh, ds.refBottomDepth, ds.minLevelCell)

    ds['restingThickness'] = compute_z_level_layer_thickness(
        ds.refTopDepth, ds.refBottomDepth, restingSSH, ds.bottomDepth,
        ds.minLevelCell, ds.maxLevelCell)

    ds['layerThickness'] = _compute_z_star_layer_thickness(
        ds.restingThickness, ds.ssh, ds.bottomDepth, ds.minLevelCell,
        ds.maxLevelCell)

    haney_cell, haney_edge = compute_haney_number(
        ds, ds.layerThickness, ds.ssh, show_progress=True)

    # --- Drop layers to stay within slope and thickness bounds ---
    nEdges = ds.sizes['nEdges']
    nCells = ds.sizes['nCells']
    nVertLevels = ds.sizes['nVertLevels']

    xCell = ds.xCell.values
    yCell = ds.yCell.values

    minLevelCell = ds.minLevelCell - 1
    maxLevelCell = ds.maxLevelCell - 1
    edgesOnCell = ds.edgesOnCell - 1

    cellsOnEdge = ds.cellsOnEdge - 1
    internal_mask = numpy.logical_and(cellsOnEdge[:, 0] >= 0,
                                      cellsOnEdge[:, 1] >= 1)
    cell0 = cellsOnEdge[:, 0].values
    cell1 = cellsOnEdge[:, 1].values
    cell0 = cell0[internal_mask]
    cell1 = cell1[internal_mask]

    vert_index = \
        xarray.DataArray.from_dict({'dims': ('nVertLevels',),
                                    'data': numpy.arange(nVertLevels)})

    cell_mask = numpy.logical_and(vert_index >= minLevelCell,
                                  vert_index <= maxLevelCell)

    ssh = ds.ssh
    if 'Time' in ssh.dims:
        ssh = ssh.isel(Time=0)
    ssh = ssh.values
    bottomDepth = ds.bottomDepth.values

    layerThickness = ds.layerThickness.where(cell_mask, 0.).values
    nIters = 1
    dzdx_thresh = 1e2/1e4 # 100m/10km
    for iIter in range(nIters):

        # Update minLevelCell
        # NOTE currently testing with sub-ice-shelf 2d case
        #for edgeIndex,cell0Index in enumerate(cell0):
        for edgeIndex in range(len(cell0)):
            cell0Index = cell0[edgeIndex]
            cell1Index = cell1[edgeIndex]
            z0 = (  ssh[cell0Index]
                  - 0.5*layerThickness[cell0Index,minLevelCell[cell0Index]]) 
            z1 = (  ssh[cell1Index]
                  - 0.5*layerThickness[cell1Index,minLevelCell[cell1Index]]) 
            dz = z0 - z1
            dx = numpy.sqrt(  numpy.square(xCell[cell0Index]
                                         - xCell[cell1Index])
                            + numpy.square(yCell[cell0Index]
                                         - yCell[cell1Index]))
            if (dz/dx > dzdx_thresh):
               minLevelCell[cell1Index] = min(nVertLevels - 1,
                                              minLevelCell[cell1Index] + 1)
            elif (dz/dx < -1*dzdx_thresh):
               minLevelCell[cell0Index] = min(nVertLevels - 1,
                                              minLevelCell[cell0Index] + 1)

        # Update maxLevelCell
        # TODO needs to be tested with sloped bed
        for edgeIndex in range(len(cell0)):
            cell0Index = cell0[edgeIndex]
            cell1Index = cell1[edgeIndex]
            z0 = ( -1.0*bottomDepth[cell0Index]
                  + 0.5*layerThickness[cell0Index,maxLevelCell[cell0Index]]) 
            z1 = ( -1.0*bottomDepth[cell1Index]
                  + 0.5*layerThickness[cell1Index,maxLevelCell[cell1Index]]) 
            dz = z0 - z1
            dx = numpy.sqrt(  numpy.square(xCell[cell0Index]
                                         - xCell[cell1Index])
                            + numpy.square(yCell[cell0Index]
                                         - yCell[cell1Index]))
            if (dz/dx > dzdx_thresh):
               maxLevelCell[cell0Index] = max(0, maxLevelCell[cell0Index] - 1)
            elif (dz/dx < -1*dzdx_thresh):
               maxLevelCell[cell1Index] = max(0, maxLevelCell[cell1Index] - 1)

        ds['layerThickness'] = _compute_z_star_layer_thickness(
            ds.restingThickness, ds.ssh, ds.bottomDepth, ds.minLevelCell,
            ds.maxLevelCell)

        cell_mask = numpy.logical_and(vert_index >= minLevelCell,
                                      vert_index <= maxLevelCell)
        layerThickness = ds.layerThickness.where(cell_mask, 0.).values

    haney_cell, haney_edge = compute_haney_number(
        ds, ds.layerThickness, ds.ssh, show_progress=True)

    

def _compute_z_star_layer_thickness(restingThickness, ssh, bottomDepth,
                                    minLevelCell, maxLevelCell):
    """
    Compute z-star layer thickness by stretching restingThickness based on ssh
    and bottomDepth

    Parameters
    ----------
    restingThickness : xarray.DataArray
        The thickness of z-star layers when ssh = 0

    ssh : xarray.DataArray
        The sea surface height

    bottomDepth : xarray.DataArray
        The positive-down depth of the seafloor

    minLevelCell : xarray.DataArray
        The zero-based index of the top valid level

    maxLevelCell : xarray.DataArray
        The zero-based index of the bottom valid level

    Returns
    -------
    layerThickness : xarray.DataArray
        The thickness of each layer (level)
    """

    nVertLevels = restingThickness.sizes['nVertLevels']
    refThickness = []
    layerThickness = []

    for zIndex in range(nVertLevels):
        mask = numpy.logical_and(zIndex >= minLevelCell,
                                 zIndex <= maxLevelCell)
        thickness = restingThickness.isel(nVertLevels=zIndex)
        thickness = thickness.where(mask, 0.)
        refThickness.append(thickness)

    # layers should be stretched according to 
    # sum_k(restingThickness(k)),k=minLevelCell,maxLevelCell 
    # which can be != bottomDepth in contrast to previous implementation
    refThickness = xarray.concat(refThickness, dim='nVertLevels')
    H_new = refThickness.sum(dim='nVertLevels')

    layerStretch = (ssh + bottomDepth) / H_new

    for zIndex in range(nVertLevels):
        thickness = layerStretch*restingThickness.isel(nVertLevels=zIndex)
        thickness = thickness.where(mask, 0.)
        layerThickness.append(thickness)
    layerThickness = xarray.concat(layerThickness, dim='nVertLevels')
    layerThickness = layerThickness.transpose('nCells', 'nVertLevels')

    return layerThickness


