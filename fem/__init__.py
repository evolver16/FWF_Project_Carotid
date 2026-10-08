"""Total-Lagrangian FE framework, independent of the G&R models: core (assembly, Newton, time loop, buckling),
elements, mesh (generators, Abaqus .inp, VTK output), tensor3 (3x3 helpers), pytree (state classes);
System(mesh) without a model: plain materials"""

import jax

jax.config.update("jax_enable_x64", True)

from fem.core import (ELEMENTS, BC, Elastic, System, anderson, broadcast_state, gauss_points, geometry,
                      laplace, pardiso_available, transfer, wall_basis)
