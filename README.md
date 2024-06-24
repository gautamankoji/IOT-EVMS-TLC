# IoT-Based Intelligent Emergency Vehicle Monitoring System with Traffic Light Control (IOT-EVMS-TLC)

## Overview

This project aims to develop an IoT-based intelligent system for monitoring emergency vehicles and controlling traffic lights to facilitate their movement. The system uses machine learning models to identify emergency vehicles from video feeds, integrates with IoT devices for real-time traffic light control, and provides a smart solution for improving emergency response times.

## Table of Contents

- [IoT-Based Intelligent Emergency Vehicle Monitoring System with Traffic Light Control (IOT-EVMS-TLC)](#iot-based-intelligent-emergency-vehicle-monitoring-system-with-traffic-light-control-iot-evms-tlc)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [System Architecture](#system-architecture)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Data](#data)
  - [Model Training](#model-training)
  - [Traffic Light Control](#traffic-light-control)
  - [References](#references)

## Introduction

Emergency vehicles often face delays due to traffic congestion. This project aims to alleviate this problem by implementing a smart monitoring system that identifies emergency vehicles and dynamically controls traffic lights to provide a clear path for these vehicles.

## System Architecture

The system consists of the following components:

1. **Video Surveillance Module**: Captures real-time video feeds from traffic cameras.
2. **Vehicle Identification Module**: Uses a Convolutional Neural Network (CNN) to identify emergency vehicles.
3. **IoT Module**: Communicates with traffic light controllers to change signals based on the presence of emergency vehicles.
4. **Control Center**: A central hub that processes data and sends commands to the IoT module.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/gautamankoji/IOT-EVMS-TLC.git
   cd IOT-EVMS-TLC
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Environment**:
   - Ensure you have a working camera feed.
   - Configure your IoT devices and traffic light controllers.

## Usage

1. **Run the Vehicle Identification Script**:
   ```bash
   python vehicle_detection.py
   ```

2. **Monitor the Output**:
   - The script will process video feeds and identify emergency vehicles.
   - Traffic light control signals will be sent to the IoT module.

## Data

The dataset used for training the vehicle identification model can be found at [Kaggle](https://www.kaggle.com/datasets/abhisheksinghblr/emergency-vehicles-identification/data). It contains images of various vehicles, including emergency vehicles, which are used to train the CNN model.

## Model Training

The model is trained using a Convolutional Neural Network (CNN) implemented with Keras. The steps to train the model are outlined in the `vehicle_detection.py` script. The trained model is saved as `vehicle.h5`.

## Traffic Light Control

The traffic light control logic is implemented in the IoT module. When an emergency vehicle is identified, the system sends signals to the traffic light controllers to change the lights, providing a clear path for the emergency vehicle.

## References

- [Emergency Vehicles Identification Dataset](https://www.kaggle.com/datasets/abhisheksinghblr/emergency-vehicles-identification/data)
- [IoT-Based Smart Ambulance Monitoring System with Traffic Light Control](https://www.slideshare.net/slideshow/iot-based-smart-ambulance-monitoring-system-with-traffic-light-control/253637670)
- [IEEE Paper on Smart Traffic Light Control System](https://ieeexplore.ieee.org/document/10112694)
