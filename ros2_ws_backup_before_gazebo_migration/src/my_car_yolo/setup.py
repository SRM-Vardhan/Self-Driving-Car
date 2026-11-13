from setuptools import find_packages, setup

package_name = 'my_car_yolo'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),
        (
            'share/' + package_name,
            ['package.xml']
        ),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kunal-singh',
    maintainer_email='singhk122007@gmail.com',
    description='ROS 2 interface node for future YOLO-based obstacle detection',
    license='TODO',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolo_node = my_car_yolo.yolo_node:main',
        ],
    },
)