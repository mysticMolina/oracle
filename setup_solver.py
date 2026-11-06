#!/usr/bin/env python3
"""Install a local Python environment, run checks, or execute the small example."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import venv
ROOT=Path(__file__).resolve().parent
ENV=ROOT/'.venv'
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',nargs='?',default='install',choices=('install','test','demo'))
    p.add_argument('--yes',action='store_true')
    a=p.parse_args();python=ENV/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    try:
        if a.command=='install':
            if sys.version_info<(3,10):raise ValueError('Python 3.10+ is required')
            if not a.yes:
                print('Install solver, plots and tests in this checkout\'s .venv. No MATLAB or shell changes.')
                if input('Continue? [Y/n] ').strip().lower() not in ('','y','yes'):return 0
            if ENV.exists() and not (ENV/'pyvenv.cfg').is_file():raise ValueError('Existing .venv is not a virtual environment')
            if not ENV.exists():venv.EnvBuilder(with_pip=True).create(ENV)
            subprocess.run([str(python),'-m','pip','install','-e',str(ROOT)+'[plots,test]'],check=True,cwd=ROOT)
            print('Installed. Next: python setup_solver.py test; python setup_solver.py demo')
        else:
            if not python.is_file():raise ValueError('Run python setup_solver.py install first')
            command=[str(python),'-m','pytest','tests','-q'] if a.command=='test' else [str(python),str(ROOT/'examples/quickstart.py')]
            subprocess.run(command,check=True,cwd=ROOT)
        return 0
    except (ValueError,OSError,subprocess.CalledProcessError) as exc:
        print('error:',exc,file=sys.stderr);return 1
if __name__=='__main__':raise SystemExit(main())
