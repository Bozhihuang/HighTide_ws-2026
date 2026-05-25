from setuptools import setup
package_name = 'hightide_drivers'
setup(
    name=package_name, version='1.0.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='hightide Team', maintainer_email='team@hightide.org',
    description='GPIO actuator drivers for torpedoes and marker droppers',
    license='MIT',
    entry_points={'console_scripts': [
        'actuator_driver_node = hightide_drivers.actuator_driver_node:main',
    ]},
)
