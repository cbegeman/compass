from compass.ocean.tests.drying_slope.forward import Forward
from compass.ocean.tests.drying_slope.initial_state import InitialState
from compass.testcase import TestCase
from compass.validate import compare_variables


class Decomp(TestCase):
    """
    A decomposition test case for the baroclinic channel test group, which
    makes sure the model produces identical results on 1 and 12 cores.

    Attributes
    ----------
    resolution : str
        The resolution of the test case

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

        method : str
            The type of wetting-and-drying algorithm

        time_integrator : {'rk4', 'split_explicit'}, str
            The time integration scheme to use for this test case
        """
        name = 'decomp'
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

        if coord_type == 'single_layer':
            damping_coeff = None
        else:
            damping_coeff = 0.01
        for procs in [1, 12]:
            name = '{}proc'.format(procs)
            forward_step = Forward(test_case=self, name=name, subdir=name,
                                   resolution=resolution,
                                   ntasks=procs, openmp_threads=1,
                                   damping_coeff=damping_coeff,
                                   coord_type=coord_type,
                                   time_integrator=time_integrator)
            if method == 'ramp':
                forward_step.add_namelist_options(
                    {'config_zero_drying_velocity_ramp': ".true."})
            self.add_step(forward_step)

    def validate(self):
        """
        Test cases can override this method to perform validation of variables
        and timers
        """
        variables = ['temperature', 'salinity', 'layerThickness',
                     'normalVelocity']
        compare_variables(test_case=self, variables=variables,
                          filename1='1proc/output.nc',
                          filename2='12proc/output.nc')

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
