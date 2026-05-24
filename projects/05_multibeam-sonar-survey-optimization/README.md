# Multibeam Sonar Survey Line Optimization

## Overview

This mathematical modeling project optimizes survey line design for multibeam sonar systems under realistic seabed coverage constraints.

## Research Question

How can survey lines be designed to minimize total path length while satisfying seabed coverage and overlap-rate requirements?

## Methods

- Modeled multibeam sonar coverage width and overlap rate using geometric relationships.
- Extended the model from 2D seabed sections to 3D seabed surfaces.
- Used Python iteration for coverage-width calculations under different survey directions and ship positions.
- Used MATLAB local search and linear interpolation to optimize survey line layout on a known-depth sea area.
- Evaluated total survey length, missed coverage area, and excessive overlap.

## Results

- Built a full modeling pipeline from idealized geometry to practical survey-line optimization.
- One reported optimization comparison showed 52-54 survey lines under different step sizes, with total lengths ranging from 397,875 m to 488,150 m.
- The model balanced coverage completeness, overlap constraints, and route length.

## Files

- `report.pdf`: Full mathematical modeling report.

## Resume Bullet

Designed a multibeam sonar survey-line optimization model using 2D/3D geometric derivation, local search, and interpolation to minimize route length under coverage and overlap constraints.
