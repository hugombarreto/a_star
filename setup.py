from distutils.core import setup

setup(
    name='a_star',
    version='0.95a',

    packages=[
        'a_star',
    ],

    package_data={
        'a_star': ['defaults.cfg'],
    },

    install_requires=[
        'ctree',
    ]
)

