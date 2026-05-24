from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'HighTide_tests'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='HighTide Team',
    maintainer_email='team@HighTide.org',
    description='Unit and integration tests for all HighTide packages',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'pool_test_actuators = HighTide_tests.pool_tests.pool_test_actuators:main',
        'pool_test_depth = HighTide_tests.pool_tests.pool_test_depth:main',
        'pool_test_thrusters = HighTide_tests.pool_tests.pool_test_thrusters:main',
        'pool_test_sensors = HighTide_tests.pool_tests.pool_test_sensors:main',
        'pool_test_navigation = HighTide_tests.pool_tests.pool_test_navigation:main',
    ]},
)
