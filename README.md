# Autonomous-Satellite-Collision-Avoidance-System
Autonomous satellite collision avoidance simulator using poliastro, astropy, numpy, and scipy. The system propagates satellite and debris orbits, detects conjunctions, predicts time of closest approach, and performs avoidance maneuvers. It can also generate synthetic datasets for ML-based space traffic management research.

Autonomous Satellite Collision Avoidance Simulator
=================================================

Overview
--------
This project is a Python-based orbital simulation and collision avoidance system built to model satellite-debris conjunction scenarios in Low Earth Orbit (LEO). It combines orbital propagation, synthetic debris generation, conjunction detection, closest-approach analysis, avoidance maneuver simulation, visualization, and dataset generation in a modular workflow.

The main goal of the project is to demonstrate an end-to-end autonomous collision-avoidance pipeline for satellites using realistic orbital mechanics foundations. It is intended as both a technical portfolio project and a foundation for future extensions such as machine learning-based risk prediction, maneuver optimization, and higher-fidelity astrodynamics modeling.

What This Project Does
----------------------
The simulator currently supports:

- creation of a satellite orbit
- generation of a synthetic debris population
- numerical propagation of satellite and debris trajectories
- J2 perturbation modeling (Earth oblateness / orbital precession effects)
- conjunction detection using configurable distance thresholds
- time of closest approach (TCA) and minimum separation estimation
- simple avoidance maneuver execution using delta-v changes
- 3D visualization of Earth, orbit tracks, debris cloud, and maneuver locations
- generation of labeled CSV datasets for future machine learning experiments

Project Structure
-----------------
Main Python modules:

- orbit_propagator.py
  Creates the satellite orbit and handles propagation physics.
  This now includes a numerical propagator with optional J2 perturbation.

- debris_generator.py
  Generates randomized debris orbits using orbital elements such as altitude,
  inclination, RAAN, argument of perigee, true anomaly, and small eccentricity.

- conjunction_detection.py
  Computes distances and identifies conjunction candidates within a threshold.

- closest_approach.py
  Estimates time of closest approach and minimum separation distance over a time window.

- satellite_maneuver.py
  Applies a simple maneuver by modifying the satellite velocity vector.

- simulation_engine.py
  Runs the main simulation loop:
  propagation -> conjunction detection -> closest approach -> maneuver decision -> logging

- visualization.py
  Plots Earth, orbit paths, debris cloud, conjunction/maneuver activity, and flown trajectory.

- dataset_generator.py
  Creates CSV datasets of conjunction scenarios for downstream analysis or ML work.

Notebook Files
--------------
The Jupyter notebooks are the easiest way to explore the project.

- Simulation.ipynb
- 2nd Simulation.ipynb

These notebooks contain working examples, experiments, and demonstration cells showing how to:
- generate debris
- propagate orbits
- run the simulation engine
- visualize the system
- generate conjunction datasets

If you want to quickly understand the project, start with the notebook files.

Physics Summary
---------------
The project is based on orbital mechanics in an Earth-centered inertial-style framework.

At the core, each satellite or debris object is represented by orbital state vectors / orbital elements and propagated forward in time. The simulation includes:

1. Two-body orbital motion
   The base dynamics are governed by Earth’s gravity.

2. J2 perturbation
   Earth is not a perfect sphere. Its oblateness creates the J2 perturbation, which causes orbital precession and makes the simulation more realistic than a purely ideal two-body model.

3. Relative motion / conjunction analysis
   The simulator computes distances between the main satellite and debris objects over time to identify close approaches.

4. Closest approach prediction
   For flagged conjunctions, the code estimates the time of closest approach (TCA) and minimum miss distance inside a future time window.

5. Avoidance maneuver logic
   If a predicted conjunction is severe enough, a maneuver is triggered by applying a small delta-v to the satellite.

Why This Project Was Built
--------------------------
This project was built as a portfolio-grade aerospace / astrodynamics software project to demonstrate:

- applied orbital mechanics
- scientific Python programming
- simulation system design
- collision-risk analysis
- modular engineering workflow
- readiness for future AI / ML integration

It is also meant to serve as a platform for future expansion into more advanced collision-avoidance research.

How To Use
----------
1. Install the required Python libraries:
   - numpy
   - scipy
   - astropy
   - poliastro
   - matplotlib
   - jupyter

2. Open the notebook files and run the cells.
   The notebooks provide sample usage and are the best entry point.

3. If working directly in Python scripts:
   - create a satellite orbit
   - generate debris
   - configure the simulation engine
   - run the engine
   - visualize the results
   - optionally generate a dataset

Typical workflow:
- create orbit
- run simulation
- inspect conjunctions
- inspect maneuvers
- generate plots
- export dataset

Example High-Level Workflow
---------------------------
1. Create the satellite orbit.
2. Generate a debris population.
3. Run the simulation engine with chosen thresholds and timestep.
4. Detect conjunction events.
5. Compute closest approaches.
6. Trigger avoidance maneuvers if needed.
7. Visualize the nominal and flown paths.
8. Export a conjunction dataset if desired.

Notes
-----
- Threshold selection matters. A larger conjunction threshold will produce more candidate events.
- A maneuver is only triggered if the predicted closest approach is inside the maneuver threshold.
- Because debris is randomly generated, different runs can produce different conjunction behavior.
- The CSV dataset is useful for later machine learning or statistical analysis, but the labels are still based on the assumptions of this simulator.

Future Extensions
-----------------
Possible future improvements include:
- atmospheric drag
- higher-order perturbations
- covariance-based collision probability
- optimized maneuver planning
- TLE / real catalog ingestion
- reinforcement learning or supervised ML for maneuver recommendation
- uncertainty propagation
- better mission-style reporting and analytics

Author Note
-----------
This repository is part of an ongoing autonomous satellite collision avoidance project.
A separate LaTeX PDF with the mathematical background, modeling assumptions, and additional physics details will be added later.

