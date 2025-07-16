from setuptools import setup

setup(
    name='lh_devices',
    version='0.1.0',
    packages=['lh_devices',
              'lh_devices.camera',
              'lh_devices.gilson',
              'lh_devices.hamilton',
              'lh_devices.injection',
              'lh_devices.qcmd',
              'lh_devices.rinse'],
    package_data={'lh_devices': ['templates/roadmap.html']},
    url='https://github.com/roadmap-automation/lh_devices',
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
    ],
)
