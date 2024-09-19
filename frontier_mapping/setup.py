from setuptools import find_packages, setup

package_name = 'frontier_mapping'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools','opencv-python'],
    zip_safe=True,
    maintainer='dji',
    maintainer_email='344248024@qq.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        'semantic_mapping_node = frontier_mapping.test_node:main',
        ],
    },
)
