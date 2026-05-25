from setuptools import setup
import os
from glob import glob

package_name = 'hightide_launch'
setup(
    name=package_name, version='1.0.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='hightide Team', maintainer_email='team@hightide.org',
    description='Launch files and configuration for the hightide AUV',
    license='MIT',
    entry_points={'console_scripts': []},
)
