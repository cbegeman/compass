import numpy as np
import xarray as xr
from mpas_tools.cime.constants import constants
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh

from compass.ocean.iceshelf import (
    compute_land_ice_draft_from_pressure,
    compute_land_ice_pressure_from_draft,
    compute_land_ice_pressure_from_thickness,
)
from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for ice-shelf 2D test
    cases

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """
    def __init__(self, test_case, resolution, time_varying_forcing=False):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case
        """
        super().__init__(test_case=test_case, name='initial_state')
        self.resolution = resolution
        self.time_varying_forcing = time_varying_forcing

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'initial_state.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['ice_shelf_2d']
        nx = section.getint('nx')
        ny = section.getint('ny')
        dc = section.getfloat('dc')

        dsMesh = make_planar_hex_mesh(nx=nx, ny=ny, dc=dc, nonperiodic_x=False,
                                      nonperiodic_y=True)
        write_netcdf(dsMesh, 'base_mesh.nc')

        dsMesh = cull(dsMesh, logger=logger)
        dsMesh = convert(dsMesh, graphInfoFileName='culled_graph.info',
                         logger=logger)
        write_netcdf(dsMesh, 'culled_mesh.nc')

        bottom_depth = config.getfloat('vertical_grid', 'bottom_depth')

        section = config['ice_shelf_2d']
        temperature = section.getfloat('temperature')
        surface_salinity = section.getfloat('surface_salinity')
        bottom_salinity = section.getfloat('bottom_salinity')

        # points 1 and 2 are where angles on ice shelf are located.
        # point 3 is at the surface.
        # d variables are total water-column thickness below ice shelf
        y1 = section.getfloat('y1')
        y2 = section.getfloat('y2')
        y3 = section.getfloat('y3')
        d1 = section.getfloat('y1_water_column_thickness')
        d2 = section.getfloat('y2_water_column_thickness')
        d3 = bottom_depth

        ds = dsMesh.copy()

        ds['bottomDepth'] = bottom_depth * xr.ones_like(ds.xCell)

        yCell = ds.yCell

        column_thickness = xr.where(
            yCell < y1, d1, d1 + (d2 - d1) * (yCell - y1) / (y2 - y1))
        column_thickness = xr.where(
            yCell < y2, column_thickness,
            d2 + (d3 - d2) * (yCell - y2) / (y3 - y2))
        column_thickness = xr.where(yCell < y3, column_thickness, d3)

        modify_mask = xr.where(yCell < y3, 1, 0).expand_dims(
            dim='Time', axis=0)
        land_ice_fraction = modify_mask.astype(float)
        land_ice_mask = modify_mask.copy()
        land_ice_floating_fraction = land_ice_fraction.copy()
        land_ice_floating_mask = land_ice_mask.copy()

        ocean_density = constants['SHR_CONST_RHOSW']
        land_ice_density = constants['SHR_CONST_RHOICE']
        y0_land_ice_height_above_floatation = section.getfloat(
            'y0_land_ice_height_above_floatation')
        if y0_land_ice_height_above_floatation > 0.:
            land_ice_thickness = (bottom_depth - column_thickness) * \
                ocean_density / land_ice_density
            land_ice_thickness = xr.where(
                yCell < y1,
                land_ice_thickness +
                (y1 - yCell) * y0_land_ice_height_above_floatation / y1,
                land_ice_thickness)
            land_ice_pressure = compute_land_ice_pressure_from_thickness(
                land_ice_thickness=land_ice_thickness, modify_mask=modify_mask,
                land_ice_density=land_ice_density)
            land_ice_draft = compute_land_ice_draft_from_pressure(
                land_ice_pressure=land_ice_pressure,
                modify_mask=ds.bottomDepth > 0.)
            land_ice_draft = land_ice_draft.transpose('Time', 'nCells')
            ds['ssh'] = np.maximum(land_ice_draft, -ds.bottomDepth + d1)
            column_thickness = ds.ssh + ds.bottomDepth
        else:
            ds['ssh'] = -bottom_depth + column_thickness
            land_ice_draft = ds.ssh
            land_ice_pressure = compute_land_ice_pressure_from_draft(
                land_ice_draft=land_ice_draft, modify_mask=modify_mask,
                ref_density=ocean_density)

        # set up the vertical coordinate
        init_vertical_coord(config, ds)

        salinity = surface_salinity + ((bottom_salinity - surface_salinity) *
                                       (ds.zMid / (-bottom_depth)))
        salinity, _ = xr.broadcast(salinity, ds.layerThickness)
        salinity = salinity.transpose('Time', 'nCells', 'nVertLevels')

        normalVelocity = xr.zeros_like(ds.xEdge)
        normalVelocity, _ = xr.broadcast(normalVelocity, ds.refBottomDepth)
        normalVelocity = normalVelocity.transpose('nEdges', 'nVertLevels')
        normalVelocity = normalVelocity.expand_dims(dim='Time', axis=0)

        ds['temperature'] = temperature * xr.ones_like(ds.layerThickness)
        ds['salinity'] = salinity
        ds['normalVelocity'] = normalVelocity
        ds['fCell'] = xr.zeros_like(ds.xCell)
        ds['fEdge'] = xr.zeros_like(ds.xEdge)
        ds['fVertex'] = xr.zeros_like(ds.xVertex)
        ds['modifyLandIcePressureMask'] = modify_mask
        ds['landIceFraction'] = land_ice_fraction
        ds['landIceFloatingFraction'] = land_ice_floating_fraction
        ds['landIceMask'] = land_ice_mask
        ds['landIceFloatingMask'] = land_ice_floating_mask
        ds['landIcePressure'] = land_ice_pressure
        ds['landIceDraft'] = land_ice_draft

        write_netcdf(ds, 'initial_state.nc')

        # Generate the tidal forcing dataset whether it is used or not
        ds_forcing = xr.Dataset()
        y_max = np.max(ds.yCell.values)
        ds_forcing['tidalInputMask'] = xr.where(
            ds.yCell > (y_max - 0.6 * 5.0e3), 1.0, 0.0)
        if self.time_varying_forcing:
            self._write_time_varying_forcing(ds_init=ds,
                                             ice_density=land_ice_density)
        write_netcdf(ds_forcing, 'init_mode_forcing_data.nc')

    def _write_time_varying_forcing(self, ds_init, ice_density):
        """
        Write time-varying land-ice forcing and update the initial condition
        """

        config = self.config
        dates = config.get('ice_shelf_2d_forcing', 'dates')
        dates = [date.ljust(64) for date in dates.replace(',', ' ').split()]
        scales = config.get('ice_shelf_2d_forcing', 'scales')
        scales = [float(scale) for scale in scales.replace(',', ' ').split()]

        ds_out = xr.Dataset()
        ds_out['xtime'] = ('Time', dates)
        ds_out['xtime'] = ds_out.xtime.astype('S')

        landIceDraft = list()
        landIcePressure = list()
        landIceFraction = list()
        landIceFloatingFraction = list()

        land_ice_draft = ds_init.landIceDraft
        land_ice_pressure = ds_init.landIcePressure

        for scale in scales:
            landIceDraft.append(scale * land_ice_draft)
            landIcePressure.append(scale * land_ice_pressure)
            landIceFraction.append(ds_init.landIceFraction)
            # Since floating fraction does not change, none of the thin film
            # cases allow for the area undergoing melting to change
            landIceFloatingFraction.append(ds_init.landIceFloatingFraction)

        ds_out['landIceDraftForcing'] = xr.concat(landIceDraft, 'Time')
        ds_out.landIceDraftForcing.attrs['units'] = 'm'
        ds_out.landIceDraftForcing.attrs['long_name'] = \
            'The approximate elevation of the land ice-ocean interface'
        ds_out['landIcePressureForcing'] = \
            xr.concat(landIcePressure, 'Time')
        ds_out.landIcePressureForcing.attrs['units'] = 'm'
        ds_out.landIcePressureForcing.attrs['long_name'] = \
            'Pressure from the weight of land ice at the ice-ocean interface'
        ds_out['landIceFractionForcing'] = \
            xr.concat(landIceFraction, 'Time')
        ds_out.landIceFractionForcing.attrs['long_name'] = \
            'The fraction of each cell covered by land ice'
        ds_out['landIceFloatingFractionForcing'] = \
            xr.concat(landIceFloatingFraction, 'Time')
        ds_out.landIceFloatingFractionForcing.attrs['long_name'] = \
            'The fraction of each cell covered by floating land ice'
        write_netcdf(ds_out, 'land_ice_forcing.nc')

        ds_init['landIceDraft'] = scales[0] * land_ice_draft
        ds_init['ssh'] = land_ice_draft
        ds_init['landIcePressure'] = scales[0] * land_ice_pressure
        write_netcdf(ds_init, 'initial_state.nc')
