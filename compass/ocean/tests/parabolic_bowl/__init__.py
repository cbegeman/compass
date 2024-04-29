from compass.ocean.tests.parabolic_bowl.default import Default
from compass.testgroup import TestGroup


class ParabolicBowl(TestGroup):
    """
    A test group for parabolic bowl (wetting-and-drying) test cases
    """

    def __init__(self, mpas_core):
        """
        mpas_core : compass.MpasCore
            the MPAS core that this test group belongs to
        """
        super().__init__(mpas_core=mpas_core, name='parabolic_bowl')
        for time_integrator in ['RK4', 'split_explicit']:
            for ramp_type in ['ramp', 'noramp']:
                for wetdry in ['standard', 'subgrid']:
                    self.add_test_case(
                        Default(test_group=self,
                                ramp_type=ramp_type,
                                wetdry=wetdry,
                                time_integrator=time_integrator,
                                use_lts=False))
                # note: LTS has only standard W/D
                self.add_test_case(
                    Default(test_group=self,
                            ramp_type=ramp_type,
                            wetdry='subgrid',
                            time_integrator=time_integrator,
                            use_lts=True))
