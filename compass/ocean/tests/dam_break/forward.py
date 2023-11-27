import time

from compass.model import run_model
from compass.step import Step


class Forward(Step):
    """
    A step for performing forward MPAS-Ocean runs as part of dam break
    test cases.
    """
    def __init__(self, test_case, resolution, name='forward', subdir=None,
                 ntasks=1, min_tasks=None, openmp_threads=1,
                 time_integrator='RK4', use_lts=False):
        """
        Create a new test case

        Parameters
        ----------
        test_case : compass.TestCase
            The test case this step belongs to

        resolution : str
            The resolution of the test case

        use_lts : bool
            Whether local time-stepping is used

        name : str
            the name of the test case

        subdir : str, optional
            the subdirectory for the step.  The default is ``name``

        ntasks: int, optional
            the number of tasks the step would ideally use.  If fewer tasks
            are available on the system, the step will run on all available
            tasks as long as this is not below ``min_tasks``

        min_tasks: int, optional
            the number of tasks the step requires.  If the system has fewer
            than this number of tasks, the step will fail

        openmp_threads : int, optional
            the number of threads the step will use

        time_integrator : str, optional
            the time integration scheme.  The default is ``name``
        """
        if min_tasks is None:
            min_tasks = ntasks

        super().__init__(test_case=test_case, name=name, subdir=subdir,
                         ntasks=ntasks, min_tasks=min_tasks,
                         openmp_threads=openmp_threads)

        self.resolution = resolution
        self.time_integrator = time_integrator

        self.add_namelist_file('compass.ocean.tests.dam_break',
                               'namelist.forward')
        if time_integrator == 'RK4':
            self.add_namelist_options({'config_disable_tr_all_tend': '.true.'})
        self.add_streams_file('compass.ocean.tests.dam_break',
                              'streams.forward')

        if use_lts:
            self.add_namelist_options(
                {'config_time_integrator': "'LTS'",
                 'config_dt_scaling_LTS': "4",
                 'config_number_of_time_levels': "4",
                 'config_pressure_gradient_type': "'ssh_gradient'"})

            self.add_streams_file('compass.ocean.tests.dam_break.lts',
                                  'streams.forward')
            input_path = '../lts_regions'
            self.add_input_file(filename='mesh.nc',
                                target=f'{input_path}/lts_mesh.nc')
            self.add_input_file(filename='graph.info',
                                target=f'{input_path}/lts_graph.info')
            self.add_input_file(filename='init.nc',
                                target=f'{input_path}/lts_ocean.nc')

        else:
            self.add_streams_file('compass.ocean.tests.dam_break',
                                  'streams.forward')
            input_path = '../initial_state'
            self.add_input_file(filename='mesh.nc',
                                target=f'{input_path}/culled_mesh.nc')

            self.add_input_file(filename='init.nc',
                                target=f'{input_path}/ocean.nc')

            self.add_input_file(filename='graph.info',
                                target=f'{input_path}/culled_graph.info')

        self.add_model_as_input()

        self.add_output_file(filename='output.nc')

    # no setup() is needed

    def run(self):
        """
        Run this step of the test case
        """

        options = dict()
        thin_film_thickness = self.config.getfloat('dam_break',
                                                   'thin_film_thickness')
        options['config_drying_min_cell_height'] = f"{thin_film_thickness}"
        options['config_zero_drying_velocity_ramp_hmin'] = \
            f"{thin_film_thickness}"
        options['config_zero_drying_velocity_ramp_hmax'] = \
            f"{10. * thin_film_thickness}"
        dt, dt_btr = self.get_dt()
        if self.time_integrator == 'split_explicit':
            options['config_dt'] = f"'{dt}'"
            options['config_btr_dt'] = f"'{dt_btr}'"
        else:
            options['config_dt'] = f"'{dt_btr}'"
        self.update_namelist_at_runtime(options=options)
        run_model(self)

    def get_dt(self):
        """
        Get the time step

        Returns
        -------
        dt : str
            the time step in HH:MM:SS
        """
        config = self.config
        # dt is proportional to resolution
        dt_per_m = config.getfloat('dam_break', 'dt_per_m')
        dt_btr_per_m = config.getfloat('dam_break', 'dt_btr_per_m')

        dt_float = dt_per_m * self.resolution
        dt_btr_float = dt_btr_per_m * self.resolution
        # https://stackoverflow.com/a/1384565/7728169
        dt = time.strftime('%H:%M:%S', time.gmtime(dt_float))
        dt_btr = time.strftime('%H:%M:%S', time.gmtime(dt_btr_float))
        if dt_float < 1.:
            dt = f'{dt[:-1]}{dt_float:0.3g}'
        if dt_btr_float < 1.:
            dt_btr = f'{dt_btr[:-1]}{dt_btr_float:0.3g}'

        return dt, dt_btr
