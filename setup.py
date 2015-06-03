from distutils.core import setup

setup(
    name='AStarSpecializer',
    version='0.95a',

    packages=[
        'AStarSpecializer',
    ],

    package_data={
        'AStarSpecializer': ['defaults.cfg'],
    },

    install_requires=[
        'ctree',
    ]
)

