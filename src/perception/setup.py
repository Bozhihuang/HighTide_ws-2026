from setuptools import setup
package_name = 'hightide_perception'
setup(
    name=package_name, version='1.0.0', packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'], zip_safe=True,
    maintainer='hightide Team', maintainer_email='team@hightide.org',
    description='YOLO TensorRT detection and tracking for hightide',
    license='MIT',
    entry_points={'console_scripts': [
        'yolo_detector_node = hightide_perception.yolo_detector_node:main',
        'yolo_pt_detector_node = hightide_perception.yolo_pt_detector_node:main',
        'target_tracker_node = hightide_perception.target_tracker_node:main',
        'detection_viz_node = hightide_perception.detection_viz_node:main',
        'slalom_pole_detector_node = hightide_perception.slalom_pole_detector_node:main',
        'octagon_table_detector_node = hightide_perception.octagon_table_detector_node:main',
    ]},
)
