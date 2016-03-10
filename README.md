# Rapid - Android Offloading Framework
This is part of the [Rapid Project](rapid-project.eu) and is an ongoing work. While Rapid envisions to support heteregenous devices, this is the component that enables code offloading on Android devices.  
Check the other Rapid repositories for code offloading on other platforms.

>The main goal of the RAPID framework is to allow low-power devices to support compute-intensive tasks and applications at reasonable time. Hence, RAPID aspires to enable devices such as smartphones, notebooks, tablets, wearable devices, etc. to deliver both Central Processing Unit (CPU) and GPU intensive tasks, exploiting resources of more capable devices, or even the cloud, participating in the RAPID ecosystem. In this perspective, RAPID is developing an efficient heterogeneous CPU-GPU cloud computing infrastructure, which can be used to seamlessly offload CPU-based and GPU-based (using CUDA) tasks of applications to more powerful devices over a heterogeneous network (HetNet).

## Terminology
* **User Device (UD):** is the low-power device (phone, e.g.) that will be accelerated by code offloading. In our scenario it will be a phone running Android (version 4.1+ is recommended).
* **Clone:** is a Virtual Machine (**VM**) running on virtualized software, with the same operating system as the UD. In our scenario it will be an Android-x86 instance (version 4.0+ is recommended) running on VirtualBox.
* **Acceleration Server (AS):** is an Android application that runs on the clone and is responsible for executing the offloaded code by the client.
* **Acceleration Client (AC):** is an Android library that enables code offloading on Android applications.
* **Rapid Common:** is a library that should be included by the AC project.
* **Rapid-DemoApp:** is an Android app that implements some simple applications that use code offloading.

# Installation
We have implemented two ways of deploying the framework. 
The first one is fast and is meant to be a simple setup for testing the offloading framework. 
The second one requires all components of the Rapid project to be up and running.

## Quick setup without Rapid infrastructure
Just following the simple steps described below you should be able to run the demo applications included in this repo in less than one hour.

### What you need
* A computer running any operating system able to run a virtualization software such as [VirtualBox](https://www.virtualbox.org/).
* An Android phone running Android 4.1+ (recommended).
* A WiFi router where both computer and phone are connected.
    * The phone should be able to ping the computer, as shown in the figure below.
![](https://dl.dropboxusercontent.com/u/7728796/img-quick-setup.png "An example of a simple setup.")

### Download the components
* Download the following projects from github and build them using Eclipse.
* **AccelerationServer** is an Android application and is the component that will run on the *clone*. It will be responsible for executing the offloaded code.
* **AccelerationClient** is an *Android library* that should be included by the applications that want to perform code offloading. It should also be included by the Acceleration Server.
    * This component should include the [RapidCommon](https://github.com/RapidProjectH2020/rapid-common) project.
* **Rapid-DemoApp** implements some demo apps to demonstrate how to use the offloading framework. It includes the AccelerationClient android library.

### Download and deploy Android-x86
* Download the Android-x86 iso from [here](http://www.android-x86.org/) (4.0+ is recommended).
* Install Android-x86 on a virtualization software (e.g. [VirtualBox](https://www.virtualbox.org/)) on a computer that is connected to the same router as the phone.
* After installing the VM, make sure the network interface of Android-x86 is correctly configured and is up.
    * We recommend to configure networking as _Bridged_ adapter on VirtualBox. For more information on how to install and configure Android-x86 on VirtualBox please refer to [this page](http://www.android-x86.org/documents/virtualboxhowto).
    * Get the IP address of the Android-x86 VM by accessing the _Terminal Emulator_ app on the VM and using the ```netcfg``` command.
* Make sure the Android phone and the Android-x86 VM are on the same network (e.g. they can ping each other).

### Install _AccelerationServer_ component on Android-x86
* First connect to the Android-x86 instance from your computer using ```adb connect [VM_IP]```.
* Install the AccelerationServer project on the Android-x86 *clone*. To do so, use Eclipse to launch the project as an Android app and select the Android-x86 as the target device.
* The AS will automatically start running on the VM and will start listening by default on port 4322.

### Install _Rapid-DemoApp_ on the Android phone
* Attach your phone to the computer using the usb cable.
    * Now your device should be recognized by adb. You can check it from the terminal using the ```adb devices``` command.
* Install the Rapid-DemoApp application on the phone. From Eclipse run the application as Android app and choose the phone as target device when asked where to install it.
* The app should automatically start and a graphical application will appear.
* We built a graphical interface to easily interact with the application and with the framework.
* Try to run the simple built-in applications and check the Android logcat for errors or for info messages.

![](https://dl.dropboxusercontent.com/u/7728796/rapid1.png "The first activity used to insert the IP of the VM and start the connection.")
![](https://dl.dropboxusercontent.com/u/7728796/rapid2.png "The activity with some testing applications ready to run.")

## Full setup with Rapid infrastructure
**TODO**


