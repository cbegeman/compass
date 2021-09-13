from compass.testcase import TestCase
from compass.ocean.tests.tracers_planar.initial_state import InitialState
from compass.ocean.tests.tracers_planar.forward import Forward
from compass.ocean.tests import tracers_planar


class Diffusion(TestCase):
    """
    The default test case for the tracer test group simply creates
    the mesh and initial condition, then performs a short forward run on 4
    cores.

    Attributes
    ----------
    resolution : str
        The resolution of the test case
    """

    def __init__(self, test_group, resolution):
        """
        Create the test case

        Parameters
        ----------
        test_group : compass.ocean.tests.tracers_planar.TracersPlanar
            The test group that this test case belongs to

        resolution : str
            The resolution of the test case
        """
        name = 'diffusion'
        self.resolution = resolution
        subdir = '{}/{}'.format(resolution, name)
        super().__init__(test_group=test_group, name=name,
                         subdir=subdir)

        self.add_step(
            InitialState(test_case=self, resolution=resolution))
        self.add_step(
            Forward(test_case=self, cores=4, threads=1, resolution=resolution))

    def configure(self):
        """
        Modify the configuration options for this test case.
        """
        tracers_planar.configure(self.resolution, self.config)

    # no run() is needed because we're doing the default: running all steps
