import os
from glob import glob
from setuptools import setup

package_name = 'mpc_ibvs'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='A simple velocity publisher package',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'velocity_publisher = mpc_ibvs.velocity_publisher:main',
            'mpc_detect = mpc_ibvs.mpc_detect:main',
            'mpc_control = mpc_ibvs.mpc_control:main',
            'ir_mpc_detect = mpc_ibvs.ir_mpc_detect:main',
            'ir_mpc_control = mpc_ibvs.ir_mpc_control:main',
        ],
    },
)