# Using ORACLE

Install the package and run the quickstart from the repository root. Outputs
stay under `output/` and are excluded from Git. Keep case configurations and
recorded source/runtime identities with results. Do not overwrite a run with
a different case or use a changed source tree for its checkpoint.

The public package includes Python code, operational examples, tests, small
numerical fixtures and citation metadata. It does not include MATLAB source,
private notes, derivations, production datasets or cluster login details.

MAT files in fixtures and native outputs are numerical data read/written by
SciPy, not an executable dependency on MATLAB. Fixture hashes retain numerical
provenance; their creation scripts remain in the private research repository.

ORACLE currently provides a planar, single-species Dirac-Maxwell CPU reference with Mur-1 boundaries, direct/Krylov solves and source-checked restart. GPU/distributed execution, Mur-2 and nonplanar evolution are not supplied by this package.

Tests must run without the private checkout on PYTHONPATH. Python package
source identity is distinct from scientific validation. Onboarding and test
results for this release are recorded in `VALIDATION.json` after execution.
