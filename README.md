# Liquid handling devices (`lh_devices`)
Control and visualization framework for automated liquid handling applications

## Introduction and core concepts
`lh_devices` is a set of libraries for interacting with physical **devices** or **components** commonly used in liquid handling applications, such as valve positioners, syringe pumps and flow cells. The core organizational concept is that **assemblies** of physical devices are connected via fluid paths into a **network**, where each **node** of the network corresponds to a fluid **port** on a physical device.

**Methods** involving multiple devices or assemblies can be used to operate on the devices. These automatically generate logs using the `logging` module, return any results from the method (*e.g.* a measurement), and can save the results to a SQLite3 database.

Both devices and assemblies have a built in recursive web application builder for visualization and rudimentary control, with updates performed over a SocketIO connection.

## Structure
Base (application-independent) classes are contained in the modules in the base `lh_devices` directory. Subdirectories contain application-specific code:

* `gilson`: communications with Gilson devices, specifically through the GSIOC protocol
* `hamilton`: communications specifically with Hamilton OEM syringe pumps and valves
* `injection`: the ROADMAP project multichannel injection system. Requires an active setup to run.
* `qcmd`: the ROADMAP project QCMD measurement interface.

## Getting started
Two full applications are currently included in the repo. The injection system multichannel device controls a series of Hamilton-brand OEM syringe pumps and valves. Note that this will only work with an active setup.

`python -m lh_devices.injection.app`

The QCMD application interfaces with a QCMD instrument HTTP API and is purely for measurement. It will run without an active setup but will not be able to communicate:

`python -m lh_devices.qcmd.app`
