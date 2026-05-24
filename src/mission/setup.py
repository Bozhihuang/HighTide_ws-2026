from setuptools import setup, find_packages
package_name = 'HighTide_mission'
setup(
    name=package_name, version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='HighTide Team', maintainer_email='team@HighTide.org',
    description='py_trees behavior tree for HighTide mission execution',
    license='MIT',
    entry_points={'console_scripts': [
        'mission_node = HighTide_mission.mission_node:main',
    ]},
)
