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
        compute_min_max_level_cell(ds.refTopDepth, ds.refBottomDepth,
                                   #ds.ssh,
                                   restingSSH, 
                                   ds.bottomDepth)

    ds['bottomDepth'], ds['maxLevelCell'] = alter_bottom_depth(
        config, ds.bottomDepth, ds.refBottomDepth, ds.maxLevelCell)

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

    ssh = ds.ssh
    if 'Time' in ssh.dims:
        ssh = ssh.isel(Time=0)
    ssh = ssh.values

    minLevels = 3
    minLayerThickness = 25
    nIters = 5
    dzdx_thresh = 1e2/1e4 # 100m/10km
    #dzdx_thresh = 1e3/1e4 # no violations for ice-shelf-2d

    dx = numpy.zeros(nEdges)
    for edgeIndex in range(nEdges):
        cell0Index = cellsOnEdge[edgeIndex,0].values
        cell1Index = cellsOnEdge[edgeIndex,1].values
        dx[edgeIndex] = numpy.sqrt(  numpy.square(xCell[cell0Index]
                             - xCell[cell1Index])
                + numpy.square(yCell[cell0Index]
                             - yCell[cell1Index]))
    dx[dx < 1e-3] = numpy.nan

    # Drop layers from bottom to maintain minimum layer thickness
    ds['maxLevelCell'] = _adjust_maxLevelCell(
        ds.layerThickness, ds.restingThickness, ds.ssh, ds.bottomDepth,
        minLevels, minLayerThickness, ds.minLevelCell, ds.maxLevelCell)
    ds['layerThickness'] = _compute_z_star_layer_thickness(
        ds.restingThickness, ds.ssh, ds.bottomDepth, ds.minLevelCell,
        ds.maxLevelCell)

    # Drop layers from top to maintain slope limits
    ds['minLevelCell'] = _adjust_minLevelCell(
        ds.layerThickness, ds.restingThickness, ds.ssh, ds.bottomDepth,
        minLevels, dzdx_thresh, dx, ds.minLevelCell, ds.maxLevelCell, cellsOnEdge)
    ds['layerThickness'] = _compute_z_star_layer_thickness(
        ds.restingThickness, ds.ssh, ds.bottomDepth, ds.minLevelCell,
        ds.maxLevelCell)    
                                
    H_target = (ds.ssh.values + ds.bottomDepth.values)
    H = ds.layerThickness.sum(dim='nVertLevels')
    for iCell in range(nCells):
        if abs(H[iCell] - H_target[iCell]) > 1e-5:
            print('hybrid: H = {}, H_topo = {}'.format(H[iCell],H_target[iCell]))
        #for iCell in range(nCells):
        #    if (minLevelCell[iCell] > 0):
        #        #print('ssh = {}, zMid = {}'.format(
        #        #      ssh[iCell,minLevelCell[iCell]],
        #        #      ds.zMid[iCell,minLevelCell[iCell]].values))
        #        print('h_old = {}, h_new = {}'.format(
        #              layerThickness[iCell,minLevelCell[iCell]],
        #              ds.layerThickness[iCell,minLevelCell[iCell]].values))
        #        break

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
    # layers should be stretched according to 
    # sum_k(restingThickness(k)),k=minLevelCell,maxLevelCell 
    # which can be != bottomDepth in contrast to previous implementation

    nVertLevels = restingThickness.sizes['nVertLevels']
    refThickness = []
    layerThickness = []

    for zIndex in range(nVertLevels):
        mask = numpy.logical_and(zIndex >= minLevelCell,
                                 zIndex <= maxLevelCell)
        thickness = restingThickness.isel(nVertLevels=zIndex)
        thickness = thickness.where(mask, 0.)
        refThickness.append(thickness)

    refThickness = xarray.concat(refThickness, dim='nVertLevels')
    H_new = refThickness.sum(dim='nVertLevels')
    H_target = (ssh + bottomDepth)
    layerStretch = H_target / H_new

    for zIndex in range(nVertLevels):
        mask = numpy.logical_and(zIndex >= minLevelCell,
                                 zIndex <= maxLevelCell)
        thickness = layerStretch*restingThickness.isel(nVertLevels=zIndex)
        thickness = thickness.where(mask, 0.)
        layerThickness.append(thickness)
    layerThickness = xarray.concat(layerThickness, dim='nVertLevels')
    layerThickness = layerThickness.transpose('nCells', 'nVertLevels')
    return layerThickness

def _compute_layer_slope(layerThickness, ssh, bottomDepth, dx,
                         minLevelCell, maxLevelCell, cellsOnEdge):
    nEdges = len(cellsOnEdge)
    slope_top = numpy.zeros((nEdges))
    slope_bot = numpy.zeros((nEdges))
    for edgeIndex in range(nEdges):
        cell0Index = cellsOnEdge[edgeIndex,0].values
        cell1Index = cellsOnEdge[edgeIndex,1].values
        # if the edge is on the edge of the domain, don't evaluate
        if min(minLevelCell[cell0Index].values,minLevelCell[cell0Index].values) == -1:
            continue
        if min(maxLevelCell[cell0Index].values,maxLevelCell[cell0Index].values) == -1:
            continue
        # evaluate slope of the first continuous layer across edge
        minLevelEdgeBot = max(minLevelCell[cell0Index].values,minLevelCell[cell1Index].values)
        z0 = (  ssh[cell0Index]
              - sum(layerThickness[cell0Index,minLevelCell[cell0Index].values-1:minLevelEdgeBot-1])
              - 0.5*layerThickness[cell0Index,minLevelEdgeBot-1])
        z1 = (  ssh[cell1Index]
              - sum(layerThickness[cell1Index,minLevelCell[cell1Index].values-1:minLevelEdgeBot-1])
              - 0.5*layerThickness[cell1Index,minLevelEdgeBot-1])
        slope_top[edgeIndex] = numpy.divide((z0 - z1),dx[edgeIndex])

        maxLevelEdgeTop = min(minLevelCell[cell0Index].values,minLevelCell[cell1Index].values)
        z0 = ( -bottomDepth[cell0Index]
              + sum(layerThickness[cell0Index,maxLevelEdgeTop-1:maxLevelCell[cell0Index].values-1])
              + 0.5*layerThickness[cell0Index,maxLevelEdgeTop-1])
        z1 = ( -bottomDepth[cell1Index]
              + sum(layerThickness[cell1Index,maxLevelEdgeTop-1:maxLevelCell[cell1Index].values-1])
              + 0.5*layerThickness[cell1Index,maxLevelEdgeTop-1])
        slope_bot[edgeIndex] = numpy.divide((z0 - z1),dx[edgeIndex])
    return slope_top, slope_bot

def _adjust_maxLevelCell(layerThickness, restingThickness, ssh, bottomDepth,
                         minLevels, minLayerThickness, minLevelCell,
                         maxLevelCell):
    layerThickness_min = layerThickness.min(dim="nVertLevels").values
    nCells = len(layerThickness_min)
    nLevels = maxLevelCell.values - minLevelCell.values + 1
    drop_mask = numpy.logical_and(layerThickness_min < minLayerThickness,
                                  nLevels > minLevels)
    while numpy.sum(drop_mask) > 0.:
        print('drop layers from {}/{} cells for min thickness limit'.format(
              numpy.sum(drop_mask),nCells))
        maxLevelCell[drop_mask] = maxLevelCell[drop_mask] - 1
        layerThickness = _compute_z_star_layer_thickness(
            restingThickness, ssh, bottomDepth, minLevelCell,
            maxLevelCell)
        layerThickness_min = layerThickness.min(dim="nVertLevels").values
        nLevels = maxLevelCell.values - minLevelCell.values + 1
        drop_mask = numpy.logical_and(layerThickness_min < minLayerThickness,
                                      nLevels > minLevels)
    return maxLevelCell

def _adjust_minLevelCell(layerThickness, restingThickness, ssh, bottomDepth,
                         minLevels, dzdx_thresh, dx, minLevelCell,
                         maxLevelCell, cellsOnEdge):
    # Update slopes
    nLevels = maxLevelCell.values - minLevelCell.values + 1
    slope_top, slope_bot = _compute_layer_slope(layerThickness, ssh,
        bottomDepth, dx, minLevelCell, maxLevelCell, cellsOnEdge)
    # NOTE currently testing with sub-ice-shelf 2d case
    #for edgeIndex,cell0Index in enumerate(cell0):
    # TODO need to update slopes
    slope_bool = (abs(slope_top) > dzdx_thresh)
    while numpy.sum(slope_bool) > 0.:
        print('top slope above thresh for {} edges'.format(numpy.sum(slope_bool)))
        cell0 = cellsOnEdge[slope_bool,0].values
        cell1 = cellsOnEdge[slope_bool,1].values
        dzdx = slope_top[slope_bool]
        for idx in range(len(cell0)):
            cell0Index = cell0[idx]
            cell1Index = cell1[idx]
            minLevelEdgeBot = max(minLevelCell[cell0Index].values,
                                  minLevelCell[cell1Index].values)
            # Drop a layer from the top to decrease top slope
            if (dzdx[idx] > dzdx_thresh):
               if nLevels[cell1Index] <= minLevels:
                   slope_bool[idx] = False
               minLevelCell[cell1Index] = minLevelCell[cell1Index] + 1
            elif (dzdx[idx] < -1*dzdx_thresh):
               if nLevels[cell0Index] <= minLevels:
                   slope_bool[idx] = False
               minLevelCell[cell0Index] = minLevelCell[cell0Index] + 1
        # Update slopes
        layerThickness = _compute_z_star_layer_thickness(
            restingThickness, ssh, bottomDepth, minLevelCell,
            maxLevelCell)
        slope_top, slope_bot = _compute_layer_slope(layerThickness, ssh,
            bottomDepth, dx, minLevelCell, maxLevelCell, cellsOnEdge)
        slope_bool = abs(slope_top) > dzdx_thresh
        nLevels = maxLevelCell.values - minLevelCell.values + 1

        # Update maxLevelCell
        # TODO needs to be tested with sloped bed
        # TODO need to update slopes
        #slope_bool = abs(slope_bot) > dzdx_thresh
        #print('bot slope above thresh for {} edges'.format(numpy.sum(slope_bool)))
        #cell0 = cellsOnEdge[slope_bool,0].values
        #cell1 = cellsOnEdge[slope_bool,1].values
        #dzdx = slope_bot[slope_bool]
        #for edgeIndex in range(len(cell0)):
        #    cell0Index = cell0[edgeIndex]
        #    cell1Index = cell1[edgeIndex]
        #    if (dz/dx > dzdx_thresh):
        #       maxLevelCell[cell0Index] = max(minLevels, maxLevelCell[cell0Index] - 1)
        #    elif (dz/dx < -1*dzdx_thresh):
        #       maxLevelCell[cell1Index] = max(minLevels, maxLevelCell[cell1Index] - 1)
    return minLevelCell
