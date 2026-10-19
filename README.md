# Pupper Julia Simulation

## Current learning project

We are building a small planar LEGO/Technic biped as the next MuJoCo + RL
exercise.  The first version has four powered joints (left/right hip and knee),
fixed broad feet, and is constrained to forward/backward motion so that we can
learn one difficulty at a time.

The original plan was to produce three repeatable scripted motions before RL:

1. walk forward;
2. get up when lying face-down;
3. get up when lying face-up.

Manual get-up tuning was initially very difficult, and the first 300,000-step
PPO experiments failed to produce a stand. Further manual iteration produced
a successful hard-coded `getup_prone_1` routine. We then used that routine to
warm-start PPO from the prone keyframe, with an explicit penalty for planted-
foot sway. After training, the RL policy stood successfully in 20/20 fixed-seed
tests and reduced mean planted-foot sway from 0.109 m/s for the scripted
behavior-cloning baseline to 0.058 m/s. Its final joint targets produce a
visibly different standing posture: the script supplied a successful starting
behavior, while RL found its own lower-sway way to stand. This was PPO
fine-tuning from a scripted demonstration, not learning from scratch.

The experiment notes, evaluation details, and next steps are in
`lego-biped/README.md`.

## Overview
This repository contains the Julia code that runs on the Raspberry Pi-based Pupper robot, as well as the Julia code for simulating the same code in a MuJoCo environment on a desktop computer. 

## Installation for Simulation
0. Clone this repository
```shell
git clone https://github.com/Nate711/PupperSimulation.git
```
1. Acquire a license for MuJoCo at http://mujoco.org/. You can get a free trial of the professional license for a month, or with a student account, a free year.

2. Save the license ```mjkey.txt``` somewhere and set the environment variable ```MUJOCO_KEY_PATH``` to that location. One way to set the environment variable is through your bash profile. On a mac this is done by adding the line
```
export MUJOCO_KEY_PATH=[YOUR PATH]/mjkey.txt
```
to your ~/.bash_profile.

3. Install Julia by visiting https://julialang.org/downloads/

4. Install the required Julia packages
```shell
julia install.jl
```

## Run Simulation
1. Enter the Julia REPL in the PupperJuliaSim directory.
2. Run
```julia
include("main.jl")
``` 
3. The MuJoCo simulator should then pop up in a new window with various interactive options. Press space to start the simulation.
- Click and drag with the left mouse button to orbit the camera, and with the right mouse button to pan the camera.
- To perturb the robot, double click on the body you want to perturb, then hold Control and click and drag with the mouse. Using the left mouse button will apply a rotational torque while the right button will apply a translational force.
- Press space to play or pause the simulation.
- Press backspace to kill the simulation window and then hit Control-C to interrupt the Julia simulation code.

## Installation for Raspberry Pi Robot
### Materials
- Raspberry Pi 4
- SD Card (32GB recommended)
- Raspberry Pi 4 power supply (USB-C, 5V, >=3A)
- Ethernet cable

### Steps
- Install Raspbian Buster Lite onto the Pi
    - Download https://www.raspberrypi.org/downloads/raspbian/
    - Use BalenaEtcher to flash the OS to the SD card
- Set up the Raspberry Pi
    - Before even ejecting the SD Card, follow the instructions on this repo to put the self-installing setup script on the Pi: https://github.com/stanfordroboticsclub/RPI-Setup 
    - Complete the “Actually Doing It”, “Getting Internet Access”, and “Getting Started With the Pi” sections
- Test that the Pi works and connects to the internet
- Install the PREEMPT-RT kernel onto the Pi
    - Download the kernel patch https://github.com/lemariva/RT-Tools-RPi/tree/master/preempt-rt/kernel_4_19_59-rt23-v7l%2B
    - Follow these instructions starting from “Transfer the Kernel” https://lemariva.com/blog/2019/09/raspberry-pi-4b-preempt-rt-kernel-419y-performance-test
    - Test by running in the shell:
        ```shell
        uname -r
        ```
- Install Julia
    ```bash
    sudo apt-get install julia
    ```
- Get the Pupper Code
    - Clone the Pupper repository https://github.com/Nate711/PupperSimulation
    - Enter the julia repl in shell:
        ```bash
        julia
        ```
    - Install requirements by running the installation script: 
        ```julia
        include("install_for_robot.jl")
        ```
## Running the Robot
- Start the PiGPIO daemon by executing in shell:
    ```shell
    sudo pigpiod
    ```
- Load the robot code in the Julia REPL: 
    ```julia
    include("run_robot.jl")
    ``` 
- Run the main program:
    ```julia
    main()
    ```
