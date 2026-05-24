from setuptools import setup
package_name = 'HighTide_navigation'
setup(
    name=package_name, version='1.0.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='HighTide Team', maintainer_email='team@HighTide.org',
    description='Navigation stack: waypoint nav, vision servo, dead reckoning, strafe',
    license='MIT',
    entry_points={'console_scripts': [
        'waypoint_navigator_node = HighTide_navigation.waypoint_navigator_node:main',
        'vision_servo_node = HighTide_navigation.vision_servo_node:main',
        'dead_reckoning_node = HighTide_navigation.dead_reckoning_node:main',
        'yaw_controller_node = HighTide_navigation.yaw_controller_node:main',
        'search_pattern_node = HighTide_navigation.search_pattern_node:main',
    ]},
)
