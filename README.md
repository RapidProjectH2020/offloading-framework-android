# Rapid - Android Offloading Framework
This is part of the [Rapid Project](rapid-project.eu).

### Rapid mission
>RAPID targets a novel heterogeneous CPU-GPU multilevel Cloud acceleration focusing on applications running on embedded systems of low-power devices.
>The project takes advantage of abundant mobile computation power and ubiquitous high-speed networks to provide a distributed heterogeneous acceleration infrastructure that can change the future of mobile applications.


# Installation
We have implemented two ways of deploying the framework. 
The first one is fast and is meant to be a simple setup for testing the offloading framework. 
The second one requires all components of the Rapid project to be up and running and working correctly.

## Quick setup without Rapid infrastructure
This simple setup doesn't require the DS, the VMM, or the SLAM components to be running.  
Just following the simple steps described below you should be able to run the demo applications included in this repo in less than one hour.

### Download the needed components
* **AccelerationServer** is the component that will run on the *clone* and will be responsible for executing the offloaded code.
* **AccelerationClient** is an *Android library* that should be included by the applications that want to perform code offloading and by the AccelerationServer.
    * This component should include the [RapidCommon](https://github.com/RapidProjectH2020/rapid-common) project.
* **Rapid-DemoApp** includes some demo apps to demonstrate how to use the offloading framework.

### Download Android-x86
* Download the Android-x86 iso from [here](http://www.android-x86.org/).
    * We recommend Android 4.0 or later.
* Install android-x86 on a virtualization software (e.g. [VirtualBox](https://www.virtualbox.org/)).
* Make sure the network interface of android-x86 is correctly configured and is up.
    * We recommend to configure networking as _Bridged_ adapter.
* Make sure that the testing phone (see below) and android-x86 are on the same network (e.g. they can ping each other).

### Install _AccelerationServer_ component on Android-x86
* Install the AccelerationServer project on the Android-x86 *clone*.
* The server starts listening by default on port 4322, so make sure this port is accessible.

### Install _Rapid-DemoApp_ on the Android phone
* We built a graphical interface to easily interact with the application and with the framework.
* Try to run the simple built-in applications and check the log for errors or for info messages.


## Full setup with Rapid infrastructure
**TODO**


