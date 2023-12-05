import time

from compass.ocean.iceshelf import adjust_ssh
from compass.step import Step


class SshAdjustment(Step):
    """
    A step for iteratively adjusting the pressure from the weight of the ice
    shelf to match the sea-surface height as part of ice-shelf 2D test cases
    """
    def __init__(self, test_case, resolution, vertical_coordinate='z-star',
                 thin_film_present=False):
        """
        Create the step

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : float
            The horizontal resolution (km) of the test case
        """
        super().__init__(test_case=test_case, name='ssh_adjustment',
                         ntasks=None, min_tasks=None,
                         openmp_threads=None)

        self.resolution = resolution

        # generate the namelist, replacing a few default options
        # start with the same namelist settings as the forward run
        self.add_namelist_file('compass.ocean.tests.isomip_plus',
                               'namelist.forward_and_ssh_adjust')
        if vertical_coordinate == 'single_layer':
            self.add_namelist_file(
                'compass.ocean.tests.isomip_plus',
                'namelist.single_layer.forward_and_ssh_adjust')
        if thin_film_present:
            self.add_namelist_file('compass.ocean.tests.isomip_plus',
                                   'namelist.thin_film.forward_and_ssh_adjust')

        # we don't want the global stats AM for this run
        options = dict()
        options['config_AM_globalStats_enable'] = '.false.'
        self.add_namelist_options(options)

        # we want a shorter run and no freshwater fluxes under the ice shelf
        # from these namelist options
        self.add_namelist_file('compass.ocean.namelists',
                               'namelist.ssh_adjust')

        self.add_streams_file('compass.ocean.streams', 'streams.ssh_adjust')

        self.add_input_file(filename='adjusting_init0.nc',
                            target='../initial_state/initial_state.nc')

        self.add_input_file(filename='graph.info',
                            target='../cull_mesh/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='adjusted_init.nc')

    def setup(self):
        """
        Set up the test case in the work directory, including downloading any
        dependencies
        """
        self._get_resources()

    def constrain_resources(self, available_resources):
        """
        Update resources at runtime from config options
        """
        self._get_resources()
        super().constrain_resources(available_resources)

    def run(self):
        """
        Run this step of the test case
        """
        config = self.config
        resolution = self.resolution

        time_integrator = config.get('isomip_plus', 'time_integrator')
        min_layer_thickness = config.getfloat('isomip_plus',
                                              'min_layer_thickness')
        dt_per_km = config.getfloat('isomip_plus', 'dt_per_km')
        dt_btr_per_km = config.getfloat('isomip_plus', 'dt_btr_per_km')

        # https://stackoverflow.com/a/1384565/7728169
        # Note: this will drop any fractional seconds, which is usually okay
        dt = time.strftime('%H:%M:%S', time.gmtime(dt_per_km * resolution))
        btr_dt_float = dt_btr_per_km * resolution
        btr_dt = time.strftime(
            '%H:%M:%S', time.gmtime(btr_dt_float))
        if btr_dt_float < 1.:
            btr_dt = f'{btr_dt}.{round(btr_dt_float*100)}'
        options = dict(
            config_time_integrator=f"'{time_integrator}'",
            config_dt=f"'{dt}'",
            config_btr_dt=f"'{btr_dt}'",
            config_drying_min_cell_height=f"{min_layer_thickness}",
            config_zero_drying_velocity_ramp_hmin=f"{min_layer_thickness}")
        self.update_namelist_at_runtime(options)

        iteration_count = config.getint('ssh_adjustment', 'iterations')
        adjust_ssh(variable='landIcePressure', iteration_count=iteration_count,
                   step=self)

    def _get_resources(self):
        """
        Get resources (ntasks, min_tasks, and openmp_threads) from the config
        options
        """
        config = self.config
        self.ntasks = config.getint('isomip_plus', 'forward_ntasks')
        self.min_tasks = config.getint('isomip_plus', 'forward_min_tasks')
        self.openmp_threads = config.getint('isomip_plus', 'forward_threads')
