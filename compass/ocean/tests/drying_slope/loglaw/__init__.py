from numpy import ceil

from compass.config import CompassConfigParser
from compass.ocean.tests.drying_slope.forward import Forward
from compass.ocean.tests.drying_slope.initial_state import InitialState
from compass.ocean.tests.drying_slope.viz import Viz
from compass.testcase import TestCase
from compass.validate import compare_variables


class LogLaw(TestCase):
    """
    The drying_slope test case with log-law drag

    Attributes
    ----------
    resolution : float
        The resolution of the test case in km

    coord_type : str
        The type of vertical coordinate (``sigma``, ``single_layer``, etc.)
    """

    def __init__(self, test_group, resolution, coord_type, method,
                 time_integrator):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.drying_slope.DryingSlope
            The test group that this test case belongs to

        resolution : float
            The resolution of the test case in km

        coord_type : str
            The type of vertical coordinate (``sigma``, ``single_layer``)

        time_integrator : {'rk4', 'split_explicit'}, str
            The time integration scheme to use for this test case
        """
        name = 'loglaw'

        self.resolution = resolution
        self.coord_type = coord_type
        if resolution < 1.:
            res_name = f'{int(resolution*1e3)}m'
        else:
            res_name = f'{int(resolution)}km'
        subdir = f'{coord_type}/{method}_{time_integrator}/{res_name}/{name}'
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)
        self.add_step(InitialState(test_case=self, resolution=resolution))
        config = CompassConfigParser()
        config.add_from_package('compass.ocean.tests.drying_slope',
                                'drying_slope.cfg')
        section = config['drying_slope']
        ntasks_baseline = section.getint('ntasks_baseline')
        min_tasks = section.getint('min_tasks')
        ntasks = max(min_tasks, int(ceil(ntasks_baseline / resolution**2.)))
        forward_step = Forward(test_case=self, resolution=resolution,
                               ntasks=ntasks, min_tasks=min_tasks,
                               openmp_threads=1,
                               time_integrator=time_integrator)
        forward_step.add_namelist_options(
            {'config_implicit_bottom_drag_type': "'loglaw'"})
        self.add_step(forward_step)
        self.add_step(Viz(test_case=self, damping_coeffs=None))

    def validate(self):
        """
        Validate variables against a baseline
        """
        variables = ['layerThickness', 'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='forward/output.nc')

    def configure(self):
        """
        Change config options as needed
        """
        right_bottom_depth = 10.
        ly_analysis = 25.
        y_buffer = 5.
        ly = ly_analysis + y_buffer

        self.config.set(
            'drying_slope', 'thin_film_thickness', '1.0e-3',
            comment='Thickness of each layer in the thin film region')
        self.config.set('drying_slope', 'right_bottom_depth',
                        f'{right_bottom_depth}')
        self.config.set(
            'drying_slope', 'right_tidal_height', f'{-1 * right_bottom_depth}',
            comment='Initial tidal height at the right side of the domain')
        self.config.set('vertical_grid', 'bottom_depth',
                        str(right_bottom_depth))
        self.config.set(
            'drying_slope', 'ly_analysis', f'{ly_analysis}',
            comment='Length over which wetting and drying actually occur')
        self.config.set(
            'drying_slope', 'ly', f'{ly}', comment='Domain length')
        self.config.set('vertical_grid', 'coord_type', self.coord_type)
