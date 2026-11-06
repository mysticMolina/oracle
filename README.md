# ORACLE

ORACLE is a Python-only scientific solver package, free to use under the
[Apache 2.0 license](LICENSE). No MATLAB installation or license is required.

ORACLE currently provides a planar, single-species Dirac-Maxwell CPU reference with Mur-1 boundaries, direct/Krylov solves and source-checked restart. GPU/distributed execution, Mur-2 and nonplanar evolution are not supplied by this package.

## Install and quickstart

```bash
git clone https://github.com/mysticMolina/oracle.git
cd oracle
python3 setup_solver.py
python3 setup_solver.py test
python3 setup_solver.py demo
```

On Windows, use `py -3` if necessary. The wizard creates a local `.venv` and
installs solver, plotting and test dependencies. `install --yes` runs without
questions; installation alone never starts a simulation. Python 3.10+ is
required. Activate `.venv` or use its Python executable for subsequent commands.

Alternatively, inside your own environment:

```bash
python -m pip install -e '.[plots,test]'
python -m pytest tests -q
python examples/quickstart.py
```

## Run and inspect

```bash
oracle-campaign fixtures/matched_B200_box26r.json output/example --stop-after 1 --checkpoint-every 1
oracle-campaign fixtures/matched_B200_box26r.json output/example --resume --stop-after 2 --checkpoint-every 1
```

`--stop-after` bounds the work; it does not change the case's target duration.
Do not launch the full target casually: the supplied case is a 125x125 mesh
with a 75,000-step target. The quickstart runs one accepted step only.
The bundled Python `analysis_suite.adapters.open_run` reads saved products.


[Usage and output](docs/USAGE.md) describes interfaces and validation limits.
Numerical regression fixtures are small immutable data files read by Python;
they are not MATLAB programs. Cross-language exporters and research notebooks
are not part of this distribution.

## Citation

For the SPHINX Schrodinger-Maxwell work, cite **Jacob Matthew Molina and Hong
Qin (2026), [arXiv:2602.17429](https://arxiv.org/abs/2602.17429)**.
The citation does not establish validation of every option or of the ORACLE
Dirac extension. This package contains operational documentation; extended
theory manuscripts and private research logs are maintained separately.

## Tests and limits

Run the included Python tests before relying on changed numerical code.
Small fixture/restart tests do not establish continuum convergence or
long-time radiation-reaction accuracy. Review run status and numerical screens
before interpreting saved products. Source/runtime changes require a fresh
campaign identity; preserve a frozen installation for resumable runs.
