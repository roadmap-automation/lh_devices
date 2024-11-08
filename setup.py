from setuptools import setup

setup(
    name='lh_serial_devices',
    version='0.1',
    packages=['lh_serial_devices'],
    url='https://github.com/hoogerheide/lh_serial_devices',
    license='Public Domain',
    author='David P. Hoogerheide',
    author_email='david.hoogerheide@nist.gov',
    description='Liquid handling devices and assemblies',
    requires=["aiohttp",
              "aiohttp-jinja2",
              "aioserial",
              "python-socketio",
              "numpy",
              "scipy",
              "plotly",
              "svg.py"
    ]
)
