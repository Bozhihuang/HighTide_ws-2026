from setuptools import setup
package_name = 'hightide_navigation'
setup(
    name=package_name, version='1.0.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='hightide Team', maintainer_email='team@hightide.org',
    description='Navigation stack: waypoint nav, vision servo, dead reckoning, strafe',
    license='MIT',
    entry_points={'console_scripts': [
        'waypoint_navigator_node = hightide_navigation.waypoint_navigator_node:main',
        'vision_servo_node = hightide_navigation.vision_servo_node:main',
        'dead_reckoning_node = hightide_navigation.dead_reckoning_node:main',
        'yaw_controller_node = hightide_navigation.yaw_controller_node:main',
        'search_pattern_node = hightide_navigation.search_pattern_node:main',
    ]},
)
