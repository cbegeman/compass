from compass.testgroup import TestGroup
from compass.ocean.tests.tracers_planar.diffusion import Diffusion


class TracersPlanar(TestGroup):
    """
    A test group for baroclinic channel test cases
    """
    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='tracers_planar')

        for resolution in ['10km']:
            self.add_test_case(
                Diffusion(test_group=self, resolution=resolution))


def configure(resolution, config):
    """
    Modify the configuration options for one of the baroclinic test cases

    Parameters
    ----------
    resolution : str
        The resolution of the test case

    config : configparser.ConfigParser
        Configuration options for this test case
    """
    res_params = {'10km': {'nx': 16,
                           'ny': 50,
                           'dc': 10e3},
                  '4km': {'nx': 40,
                          'ny': 126,
                          'dc': 4e3},
                  '1km': {'nx': 160,
                          'ny': 500,
                          'dc': 1e3}}

    if resolution not in res_params:
        raise ValueError('Unsupported resolution {}. Supported values are: '
                         '{}'.format(resolution, list(res_params)))
    res_params = res_params[resolution]
    for param in res_params:
        config.set('tracers_planar', param, '{}'.format(res_params[param]))
