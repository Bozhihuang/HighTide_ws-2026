from setuptools import setup
package_name = 'HighTide_perception'
setup(
    name=package_name, version='1.0.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='HighTide Team', maintainer_email='team@HighTide.org',
    description='YOLO TensorRT detection and tracking for HighTide',
    license='MIT',
    entry_points={'console_scripts': [
        'yolo_detector_node = HighTide_perception.yolo_detector_node:main',
        'target_tracker_node = HighTide_perception.target_tracker_node:main',
        'detection_viz_node = HighTide_perception.detection_viz_node:main',
    ]},
)
