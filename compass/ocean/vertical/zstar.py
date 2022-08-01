import xarray
import numpy

from compass.ocean.vertical.grid_1d import add_1d_grid
from compass.ocean.vertical.partial_cells import alter_bottom_depth
from compass.ocean.vertical.zlevel import compute_z_level_layer_thickness, \
    compute_min_max_level_cell


def init_z_star_vertical_coord(config, ds):
    """
    Create a z-star vertical coordinate based on the config options in the
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
    config : compass.config.CompassConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    ds : xarray.Dataset
        A data set containing ``bottomDepth`` and ``ssh`` variables used to
        construct the vertical coordinate
    """
    add_1d_grid(config, ds)

    ds['vertCoordMovementWeights'] = xarray.ones_like(ds.refBottomDepth)

    restingSSH = xarray.zeros_like(ds.bottomDepth)
    ds['minLevelCell'], ds['maxLevelCell'], ds['cellMask'] = \
        compute_min_max_level_cell(ds.refTopDepth, ds.refBottomDepth,
                                   restingSSH, ds.bottomDepth)

    if config.has_option('vertical_grid','min_layer_thickness'):
        minLayerThickness = config.getfloat('vertical_grid','min_layer_thickness')
        ds['maxLevelCell'] = _adjust_max_level_cell(ds.ssh, ds.bottomDepth,
            ds.minLevelCell, ds.maxLevelCell, minLayerThickness)

    ds['bottomDepth'], ds['maxLevelCell'] = alter_bottom_depth(
        config, ds.bottomDepth, ds.refBottomDepth, ds.maxLevelCell)

    ds['restingThickness'] = compute_z_level_layer_thickness(
        ds.refTopDepth, ds.refBottomDepth, restingSSH, ds.bottomDepth,
        ds.minLevelCell, ds.maxLevelCell)

    ds['layerThickness'] = _compute_z_star_layer_thickness(
        ds.restingThickness, ds.ssh, ds.bottomDepth, ds.minLevelCell,
        ds.maxLevelCell)


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
    layerThickness = []

    layerStretch = (ssh + bottomDepth) / bottomDepth
    for zIndex in range(nVertLevels):
        mask = numpy.logical_and(zIndex >= minLevelCell,
                                 zIndex <= maxLevelCell)
        thickness = layerStretch*restingThickness.isel(nVertLevels=zIndex)
        thickness = thickness.where(mask, 0.)
        layerThickness.append(thickness)
    layerThickness = xarray.concat(layerThickness, dim='nVertLevels')
    layerThickness = layerThickness.transpose('nCells', 'nVertLevels')
    columnThickness = numpy.sum(layerThickness.values,axis=1)
    return layerThickness

def _adjust_min_level_cell(ssh, bottomDepth, minLevelCell, maxLevelCell,
                           minLayerThickness):
    """
    Compute z-star layer thickness by stretching restingThickness based on ssh
    and bottomDepth

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    ssh : xarray.DataArray
        The sea surface height

    bottomDepth : xarray.DataArray
        The positive-down depth of the seafloor

    maxLevelCell : xarray.DataArray
        The zero-based index of the bottom valid level

    Returns
    -------
    minLevelCell : xarray.DataArray
        The zero-based index of the top valid level
    """
    # TODO consider adding cellMask to output
    columnThickness = bottomDepth + ssh

    minLevelCell2 = maxLevelCell - numpy.floor(
        columnThickness/minLayerThickness)

    print('Adjusted minLevelCell for n={} cells'.format(
        numpy.sum(minLevelCell.values!=minLevelCell2.values)))
    print('Zeroed minLevelCell for n={} cells'.format(
        numpy.sum(minLevelCell.values>maxLevelCell.values)))

    minLevelCell = numpy.maximum(minLevelCell,minLevelCell2)

    minLevelCell[minLevelCell>maxLevelCell] = 0

    return minLevelCell

def _adjust_max_level_cell(ssh, bottomDepth, minLevelCell, maxLevelCell,
                           minLayerThickness):
    """
    Compute z-star layer thickness by stretching restingThickness based on ssh
    and bottomDepth

    Parameters
    ----------
    config : configparser.ConfigParser
        Configuration options with parameters used to construct the vertical
        grid

    ssh : xarray.DataArray
        The sea surface height

    bottomDepth : xarray.DataArray
        The positive-down depth of the seafloor

    maxLevelCell : xarray.DataArray
        The zero-based index of the bottom valid level

    minLayerThickness: xarray.DataArray
        The minimum layer thickness

    Returns
    -------
    minLevelCell : xarray.DataArray
        The zero-based index of the top valid level
    """
    columnThickness = bottomDepth + ssh
    minLayerThickness  = max(minLayerThickness,1e-12)
    maxLevelCell2 = (minLevelCell +
        numpy.floor(columnThickness/minLayerThickness))

    print('Adjusted maxLevelCell for n={} cells'.format(
        numpy.sum(maxLevelCell.values>maxLevelCell2.values)))
    print('Zeroed maxLevelCell for n={} cells'.format(
        numpy.sum(minLevelCell.values>maxLevelCell.values)))

    maxLevelCell = numpy.minimum(maxLevelCell,maxLevelCell2)

    maxLevelCell[minLevelCell>maxLevelCell] = 0

    return maxLevelCell
