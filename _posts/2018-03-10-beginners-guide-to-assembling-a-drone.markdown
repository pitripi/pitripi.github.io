---
layout: post
title: "A Student's Guide to Assembling a Quadcopter"
description: "Assembling, breaking, fixing, breaking and reassembling a quadcopter."
date: 2018-03-10 12:00:00 +0530
image: '/images/quadcopter/quadcopter_title.jpeg'
tags:   [tech. quadcopter, drone]
---
I've spent my last six months assembling, breaking, fixing, breaking and reassembling a quadcopter. Before I started, I had assumed that the task would be easy but I had no idea about the effort it would take to make a copter fly. In this post, I share my experience which would probably save others from repeating my mistakes.

### Planning
Before purchasing the required components, it would be better to make an outline of the procedure to be followed. In my medium blog post, I listed a few problems to consider before getting started. It is a time consuming task. I spent a lot of time comparing the components before purchasing, reading documentations and watching youtube videos. I considered this as my side project and it soon turned out to be demanding.

Needless to say, this would be an expensive project requiring contuinuous purchase of broken parts. Aproaching college administration might help; my project is funded by my project mentor but it took some time to secure it.

As a beginner, it would be difficult to do all the tasks by oneself. I did not pay importance to the idea of adding more members to the team and it was probably the biggest mistake I made. Increasing team size is my current priority.

Also, the goal of the project should be figured it out as early as possible and relative research should be started. We started out with a purpose of drone delivery and aerial photography, though I don't see it completed anytime soon. There needs to be a mental preparation for facing a lot of problems. There'd be crashes, software bugs and innumerable incomprehensible hard-to-find-solutions-to problems. Sometimes, days would be spent trying to find a solution to a problem which would later turn out to be as simple as firmware update.

### Purchasing components
There are a few components that are essential for a quadcopter, rest are purpose dependent. I found [Flying Car Nanodegree's](https://in.udacity.com/course/flying-car-nanodegree--nd78) first lesson a good source for introduction. What's necessary for a quadcopter is a frame, four motors, four ESCs, a flight controller, a transmitter, a receiver, a battery and a few sensors (an accelometer, a gyroscope and a compass). Optionally, one can add a GPS module, a companion computer, a camera, etc.

![quad_comp]({{site.baseurl}}/images/quadcopter/quad_comp.jpeg)
*Quadcopter Components*

Our copter currently has following components:
* 1 x [Frame](https://robokits.co.in/drones-quad-hexa-octa-fpv/frames-misc.-parts/r450-f450-quadrotor-frame-with-integraed-pcb-for-easy-wiring)
* 4 x [Motor](https://robokits.co.in/motors/rc-brushless-motor-2212-1000kv-with-propeller-adaptor?cPath=116_118&)
* 4 x [ESC](https://robokits.co.in/motor-drives/brushless-dc/brushless-esc-30a-with-simonk-firmware-for-multirotors?cPath=116_118&)
* 4 x Propeller
* 1 x [Transmitter+Receiver](https://robokits.co.in/drones-quad-hexa-octa-fpv/radio-control-tx-rx/flysky-fs-i6-6ch-2.4ghz-transmitter-with-fs-ia6b-ppm-receiver)
* 1 x [Pixhawk](https://robokits.co.in/drones-quad-hexa-octa-fpv/flight-controllers/pixhawk-px4-2.4.8-32bit-flight-controller-w-t-switch-and-sd-card)
* 1 x [Telemetry Module](https://robokits.co.in/drones-quad-hexa-octa-fpv/fpv-video-telemetry-osd/433mhz-telemetry-module-pair-for-pixhawk-and-apm-100mw-2km-range)
* 1 x [Lipo battery](https://www.amazon.in/gp/product/B00FE0OHKK/ref=oh_aui_detailpage_o02_s00?ie=UTF8&psc=1)


### Understanding the components
I had to watch a lot of youtube videos to understand the need and the order in which the components should be assembled. This is the reason one should start early; there's going to be a lot of research before you can begin assembling your copter.

The frame holds all the components together. The copter has to be powered using a battery which would have direct connections to 4 motors (via ESCs) and a flight controller. The flight controller would manage power distribution to all other components - the transmitter, GPS module, onboard computer etc. Our frame has in-built power distribution connections. We soldered 4 ESCs to the board to attach them permanently. ESCs (Electronic Speed Controllers) are used to vary motor's speed and direction of rotation. Therefore, an ESC is needed for every motor.

A motor is fixed on each leg of the copter and connected with their respective ESCs. It is important to not attach propellers till the end for safety purpose. Controlling the speed and direction of every motor manually would be cumbersome and inefficient. Therefore, a flight controller is used for the purpose; it has connection with all ESCs.

A receiver is used to give commands to the copter from ground and a transmitter attached on the copter would receive those commands and send it to flight controller. The flight controller would then send appropriate commands to ESCs which in turn rotate the motors accordingly.

A simple command would follow the following route:\\
Transmitter --> Receiver --> Flight Controller --> ESCs -->Motors

Now, the frame has 4 motors, 4 ESCs, a flight controller, a transmitter, a receiver and a battery. We used Pixhawk as flight conmtroller which has inbuilt an accelerometer, a compass and a gyroscope, all of which have to be calibrated.


### Calibration
All the sensors need to be calibrated to give flight controller the information about frame and the components. Calibration can be easily done using [Mission Planner](http://ardupilot.org/planner/) which gives instructions for every step. You'd need to connect to a computer via USB cable though. If you don't want to connect via USB you can use radio telemetry which allows you to connect Mission Planner to your copter using radio signals. Telemetry would consist of a receiver (connected to your computer via USB) and a transmitter (connected to fight controller on the copter). Connecting to your copter via telemetry will allow you to analyze various parameters like speed, altitude, remaining, battery in Mission Planner.

We also need to bind our radio transmitter to receiver, calibrate ESCs and arm our copter before flying. All of these calibration methods will vary a little depending upon the kind of flight controller and other components you have. Reading documentation is the easiest way to figure out the steps required.

### Companion computer
For onboard processing, a companion computer is added to the copter which can Raspbery Pi, Intel Edison, Nvidia TX, etc. We connected Raspberry Pi with pixhawk using mavproxy and dronekit-python. Dronekit-python allows to make and deploy apps on the copter.

### Flying bugs
Every time you want to fly, you'd have to arm the copter which means checking all the components and starting power supply to the motors. Then, you're ready to use sticks in your transmitter to play with your copter. It sounds easy, but when we tried it out, it took us a few days to get it stable in the air. Following are the mistakes we made:
#### Wrong motor spinning direction
A quadcopter can have various frame types and each frame type would have different motor rotation configuration. In out X configutaion, diagonally opposite motors were supposed to be spinning in the same direction.
#### Using wrong propellers
There are two kinds of propellers - clockwise spinning and anticlockwise spinning and they go with their respecting motors.
#### Poor parameter tuning
PID tuning is one of the most important step in stablizing a copter. We had been ignoring this step and it resulted in random crashes.

### Status
Our quadcopter is stable in the air. We have attached Raspberry Pi, however, we are yet to give it a test fly.





