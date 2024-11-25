from setuptools import setup

setup(
    name='lh_serial_devices',
    version='0.1.0',
    packages=['lh_serial_devices',
              'lh_serial_devices.hamilton'],
    url='https://github.com/hoogerheide/lh_serial_devices',
    license='Public Domain',
    author='David P. Hoogerheide',
    author_email='david.hoogerheide@nist.gov',
    description='Liquid handling devices and assemblies',
    install_requires=["aiohttp",
              "aiohttp-jinja2",
              "aioserial",
              "python-socketio",
              "numpy",
              "scipy",
              "plotly",
              "svg.py"
    ]
)
