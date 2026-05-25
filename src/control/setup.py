from setuptools import setup

package_name = 'hightide_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hightide Team',
    maintainer_email='team@hightide.org',
    description='RC Override controller, depth controller, and mode management for ArduSub',
    license='MIT',
    entry_points={
        'console_scripts': [
            'rc_override_node = hightide_control.rc_override_node:main',
            'depth_controller_node = hightide_control.depth_controller_node:main',
            'mode_manager_node = hightide_control.mode_manager_node:main',
        ],
    },
)
