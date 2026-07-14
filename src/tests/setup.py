from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'hightide_tests'

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
    maintainer='hightide Team',
    maintainer_email='team@hightide.org',
    description='Unit and integration tests for all hightide packages',
    license='MIT',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'pool_test_actuators = hightide_tests.pool_tests.pool_test_actuators:main',
        'pool_test_depth = hightide_tests.pool_tests.pool_test_depth:main',
        'pool_test_thrusters = hightide_tests.pool_tests.pool_test_thrusters:main',
        'pool_test_sensors = hightide_tests.pool_tests.pool_test_sensors:main',
        'pool_test_navigation = hightide_tests.pool_tests.pool_test_navigation:main',
        'pool_test_barrel_roll = hightide_tests.pool_tests.pool_test_barrel_roll:main',
        'pool_test_torpedo_mission = hightide_tests.pool_tests.pool_test_torpedo_mission:main',
        'verify_model = hightide_tests.pool_tests.verify_model:main',
    ]},
)
