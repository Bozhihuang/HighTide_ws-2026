from setuptools import setup, find_packages
package_name = 'hightide_mission'
setup(
    name=package_name, version='1.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='hightide Team', maintainer_email='team@hightide.org',
    description='py_trees behavior tree for hightide mission execution',
    license='MIT',
    entry_points={'console_scripts': [
        'mission_node = hightide_mission.mission_node:main',
    ]},
)
