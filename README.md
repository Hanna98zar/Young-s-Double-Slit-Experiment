"""
# Young Double-Slit in LBM with XLB (JAX)

## Overview
This project is a computational version of the Young double-slit experiment.
A coherent wave meets a barrier with two slits.
After the slits the wave fronts overlap and an interference pattern appears.

Here we simulate a weakly compressible density disturbance.
The solver is Lattice Boltzmann Method with a D2Q9 velocity set.
The focus is wave splitting and interference with simple geometry and stable boundaries.

## Model setup
- 2D lattice domain
- Internal vertical barrier at x = barrier_x
- Two slits defined by slit_width and slit_gap
- Internal POINT source at (source_x, source_y)
- Outer boundaries that let disturbances leave the domain

## Wave source
The POINT source is implemented by overwriting f at one lattice cell.
At each timestep:
- rho(t) = rho0 + rho_amp * sin(2*pi*freq*t)
- feq(t) = QuadraticEquilibrium(rho(t), u=(0,0))
- set f[:, source_x, source_y] = feq(t)

This generates a periodic density perturbation that propagates as a wave.

## Boundary conditions
- Barrier: HalfwayBounceBackBC
  The barrier acts as a solid wall for the LBM populations.
  The slits are the barrier cells that are removed.
- Outer boundaries: ExtrapolationOutflowBC on all sides
  This reduces reflections and approximates an open domain.

## Outputs
- PNG frames of rho - rho0 using fixed scale mapping
- Optional VTK fields if save_fields_vtk is available: rho, ux, uy
- One GIF assembled from the PNG frames

## How to read the results
Before the barrier you see expanding circular wave fronts from the POINT source.
At the barrier the wave passes only through the two slits.
After the slits two wave trains overlap and form an interference pattern.
Bright and dark bands correspond to constructive and destructive interference
in the density perturbation field.

## Parameters to explore
- slit_width and slit_gap control the slit geometry
- freq controls the wavelength
- source location controls near field behavior

## Technical stack
- XLB
- JAX backend
- IncompressibleNavierStokesStepper with BGK collision
- QuadraticEquilibrium for source injection
- ExtrapolationOutflowBC and HalfwayBounceBackBC for boundaries
"""
