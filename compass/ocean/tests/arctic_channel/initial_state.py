import gsw
import numpy as np
import pandas as pd
import xarray as xr
from mpas_tools.io import write_netcdf
from mpas_tools.mesh.conversion import convert, cull
from mpas_tools.planar_hex import make_planar_hex_mesh
from scipy.interpolate import interp1d

from compass.ocean.vertical import init_vertical_coord
from compass.step import Step


class InitialState(Step):
    """
    A step for creating a mesh and initial condition for the
    arctic channel test case
    """
    def __init__(self, test_case):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
        """
        super().__init__(test_case=test_case, name='initial_state')

        self.add_input_file(filename='itp77.csv', target='itp77.csv',
                            database='mixed_layer_restrat')

        for file in ['base_mesh.nc', 'culled_mesh.nc', 'culled_graph.info',
                     'initial_state.nc']:
            self.add_output_file(file)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        logger = self.logger

        section = config['horizontal_grid']
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

        section = config['vertical_grid']
        maxDepth = section.getfloat('bottom_depth')
        nVertLevels = section.getint('vert_levels')

        section = config['arctic_channel']
        linear_Tref = section.getfloat('eos_linear_Tref')
        linear_Sref = section.getfloat('eos_linear_Sref')
        linear_densityref = section.getfloat(
            'eos_linear_densityref')
        deltaT = section.getfloat('deltaT')
        deltaS = section.getfloat('deltaS')
        zm = section.getfloat('zm')
        deltaH = maxDepth / nVertLevels

        ds = dsMesh.copy()
        nCells = ds.nCells.size
        nEdges = ds.nEdges.size
        nVertices = ds.nVertices.size

        yCell = ds.yCell.values
        angleEdge = ds.angleEdge
        cellsOnEdge = ds.cellsOnEdge.values

        ds['bottomDepth'] = maxDepth * xr.ones_like(ds.xCell)
        ssh = xr.zeros_like(ds.xCell)
        ds['ssh'] = ssh

        init_vertical_coord(config, ds)
        dsMesh['maxLevelCell'] = ds.maxLevelCell

        # alpha, beta
        alpha = linear_densityref * gsw.alpha(linear_Sref, linear_Tref, 10.0)
        beta = linear_densityref * gsw.beta(linear_Sref, linear_Tref, 10.0)

        # initial salinity, density, temperature
        data = pd.read_csv('itp77.csv', header=0, names=['z', 'p', 'pt', 'sa'])

        refZMid = ds.refZMid.values
        T_itp77 = xr.ones_like(ds.refZMid)
        S_itp77 = xr.ones_like(ds.refZMid)
        z_itp = data.z.to_numpy()
        z_itp = np.insert(z_itp, 0, 0)
        pt_itp = data.pt.to_numpy()
        pt_itp = np.insert(pt_itp, 0, pt_itp[0])
        sa_itp = data.sa.to_numpy()
        sa_itp = np.insert(sa_itp, 0, sa_itp[0])
        fT = interp1d(z_itp, pt_itp)
        fS = interp1d(z_itp, sa_itp)
        T_itp77 = fT(ds.refZMid.values)
        S_itp77 = fS(ds.refZMid.values)

        # set the mixed layer to the value at the base of the mixed layer
        T_itp77[refZMid >= -zm] = fT(-zm)
        S_itp77[refZMid >= -zm] = fS(-zm)

        beta_0 = 1.e-3
        rho_ref = 1.e3

        deltaSy = \
            (beta_0 * deltaS) / (beta / rho_ref) + \
            (alpha * deltaT / beta)
        deltaSz = 0.5 * deltaSy
        deltaTz = 0.5 * deltaT

        # Solve for the function that controls how T,S vary in the y-dimension
        Ly = ny * dc
        Ly_actual = np.max(yCell) - np.min(yCell)
        yCellNorm = yCell - 0.5 * Ly_actual
        B = Ly / (2. * np.pi)
        # This is what the text says it's implementing
        # Y = 0.5 * np.sin(yCellNorm / B)
        # This is what is shown in the paper figures
        Y = np.sin(yCellNorm / B)

        # Solve for the function that controls how T,S vary in the z-dimension
        Z = np.where(-refZMid < zm,
                     0.5 * (1. + np.tanh((refZMid + zm) / deltaH)),
                     0.)
        fz, fy = np.meshgrid(Z, Y)
        temperature = xr.ones_like(ds.zMid)
        salinity = xr.ones_like(ds.zMid)
        temperature = np.multiply(temperature, T_itp77)
        salinity = np.multiply(salinity, S_itp77)
        temperature += fz * (fy * deltaT - deltaTz)
        salinity += fz * (fy * deltaSy - deltaSz)
        ds['temperature'] = temperature
        ds['salinity'] = salinity

        # Coriolis parameter
        ds['fCell'] = (('nCells', 'nVertLevels',),
                       0.00014 * np.ones([nCells, nVertLevels]))
        ds['fEdge'] = (('nEdges', 'nVertLevels',),
                       0.00014 * np.ones([nEdges, nVertLevels]))
        ds['fVertex'] = (('nVertices', 'nVertLevels',),
                         np.zeros([nVertices, nVertLevels]))

        density = linear_densityref + beta * salinity[0, :, :] - \
            alpha * temperature[0, :, :]

        # initial velocity on edges
        u_cell = np.zeros([nCells, nVertLevels])
        normalVelocity = xr.DataArray(np.zeros([nEdges, nVertLevels]),
                                      dims=['nEdges', 'nVertLevels'])
        u = normalVelocity.copy()
        drhoy = np.zeros([nCells, nVertLevels])
        integral_drho = np.zeros([nCells])
        layerThickness = ds.layerThickness.values
        deltaRhoy = beta * deltaSy - alpha * deltaT
        g = 9.81

        for k in range(nVertLevels - 1, -1, -1):
            if (-refZMid[k] < zm):
                # d/dy(0.5 * deltaRhoy * Z * np.sin(yCellNorm / B))
                drhoy[:, k] = 0.5 * deltaRhoy * Z[k] * \
                    np.cos((yCellNorm / B)) / B
                integral_drho += \
                    layerThickness[0, :, k] * drhoy[:, k] / density[:, k]
                u_cell[:, k] = g * integral_drho / ds.fCell[:, k]
        for iEdge in range(0, nEdges):
            cell1 = cellsOnEdge[iEdge, 0] - 1
            cell2 = cellsOnEdge[iEdge, 1] - 1
            u[iEdge, :] = 0.5 * (u_cell[cell1, :] + u_cell[cell2, :])
        normalVelocity = xr.DataArray(u, dims=['nEdges', 'nVertLevels']) * \
            np.cos(angleEdge)
        ds['normalVelocity'] = normalVelocity.expand_dims(dim='Time', axis=0)
        ds['velocityMeridional'] = u.expand_dims(dim='Time', axis=0)

        rho_mid = xr.zeros_like(ds.refZMid)
        idx_min = np.argmin(yCell)
        idx_max = np.argmax(yCell)
        rho_mid = 0.5 * (density[idx_min, :] + density[idx_max, :])
        for k in range(nVertLevels - 1, -1, -1):
            ssh += layerThickness[0, :, k] * (density[:, k] - rho_mid[k])
        ssh = ssh / density[:, 0]
        ds['ssh'] = ssh.expand_dims(dim="Time", axis=0)

        # Add noise to the salinity field
        ds['salinity'] = ds.salinity + \
            1.e-3 * np.random.rand(1, nCells, nVertLevels)

        # We need to run this again because the layer thicknesses need to be
        # updated to be consistent with ssh
        init_vertical_coord(config, ds)

        write_netcdf(ds, 'initial_state.nc')
