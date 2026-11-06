"""Jacob Matthew Molina. Inception: 2026-11-05. ORACLE Python port."""

from .core import dirac_matrices, pack_field, rest_energy_step, unpack_field
from .matter import charge_density, dirac_current, full_cayley_schur_step, full_matter_step
from .maxwell import (MurContext, MurHistory, gauss_project, initialize_mur,
                      longitudinal_field, mur_context, mur_field_step, mur_reference_field)
from .evolution import State, coupled_mur_step
from .diagnostics import diagnostics
from .initialization import MatchedProblem, matched_problem, prepare_matched_campaign
from .operators import (centered_operators, compatible_operators,
                        free_dirac_block, spatial_hamiltonian)

__all__ = ["dirac_matrices", "pack_field", "rest_energy_step", "unpack_field",
           "full_cayley_schur_step", "centered_operators", "compatible_operators",
           "free_dirac_block", "spatial_hamiltonian", "charge_density", "dirac_current",
           "full_matter_step", "MurContext", "MurHistory", "gauss_project", "initialize_mur",
           "longitudinal_field", "mur_context", "mur_field_step", "mur_reference_field",
           "State", "coupled_mur_step", "diagnostics", "MatchedProblem",
           "matched_problem", "prepare_matched_campaign"]
