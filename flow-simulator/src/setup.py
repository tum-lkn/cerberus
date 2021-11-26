from setuptools import setup

setup(
    name='mydcflowsim',
    version='1.0.0',
    packages=[
        '',
        'dcflowsim.algorithm',
        'dcflowsim.algorithm.learning',
        'dcflowsim.control',
        'dcflowsim.data_writer',
        'dcflowsim.environment',
        'dcflowsim.flow_generation',
        'dcflowsim.flow_generation.data',
        'dcflowsim.mip',
        'dcflowsim.network',
        'dcflowsim.network.optical_circuit_switch',
        'dcflowsim.simulation',
        'dcflowsim.statistic_collector',

    ],
    # package_dir={'': 'src'},
    url='',
    license='',
    author='Andreas Blenk, Johannes Zerwas',
    author_email='',
    description=''
)
