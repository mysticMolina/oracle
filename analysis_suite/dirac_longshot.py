"""Jacob Matthew Molina, 2026-10-15. Genuine 2D Dirac campaign adapter.

Keep the spinor and centered projected operator distinct from scalar SPHINX.
MATLAB stores component columns with x fastest. Dz=0, no z-face projection.
"""
import json,csv
import numpy as np
import h5py
from scipy.io import loadmat
from .adapters import OracleCoupledRun,_files

class DiracLongshotRun(OracleCoupledRun):
    def __init__(self,source,allow_partial=False):
        self.source=source.resolve();cfg=loadmat(source/'configuration.mat',simplify_cells=True)
        self.sim=cfg['sim'];self.meta=cfg['meta'];self.kind='oracle_longshot';self.name=source.name
        self.steps,self.files=_files(source,'state','mat');self.times=self.steps*float(self.sim['dt'])
        status=json.loads((source/'status.json').read_text()) if (source/'status.json').exists() else {}
        self.status='completed' if status.get('status')=='completed' and self.steps[-1]==int(self.sim['steps']) else 'partial'
        if self.status!='completed' and not allow_partial:raise ValueError('Partial transfer')
        self.slice_z=0;self.physics=self.sim['physics'];g=np.asarray(self.sim['grid'],int)
        self.x=float(self.sim['origin'][0])+np.arange(g[0])*self.sim['spacing'][0]
        self.y=float(self.sim['origin'][1])+np.arange(g[1])*self.sim['spacing'][1]
        self.plot_x=self.x;self.plot_y=self.y;self.orbit_radius_a0=None
        self.notes=['Full z-invariant 2D coupled branch, per unit z length; no 3D slice.',
                    'Probability current c psi^dagger alpha psi; multiply by q for charge current.',
                    'Magnetic perturbation is B(t)-B(0), including background/boundary evolution.']
        self.diagnostics={}
        with (source/'coupled.csv').open() as f:rows=list(csv.DictReader(f))
        rows={int(float(row['step'])):row for row in rows};rows=[rows[k] for k in sorted(rows)]
        self.saved={k:np.array([float(row[k]) for row in rows]) for k in rows[0]}
        self.available=('density','current','electric','magnetic_z','magnetic_delta','poynting','em_energy',
         'component_2','component_3','ham_dirac_rest','ham_dirac_kinetic','ham_dirac_gauge',
         'ham_electric','ham_magnetic','ham_total','real_2','imag_2','real_3','imag_3','relative_phase','spin_z')
        self._initial_b=super().frame(0).images['magnetic_z'].copy()
    def load_snapshot(self,index,branch=0):
        g=tuple(np.asarray(self.sim['grid'],int));out={'step':int(self.steps[index])}
        with h5py.File(self.files[index]) as f:
            state=f[f['states'][()].ravel()[branch]]
            for k in ('psi','A','Y'):
                v=state[k][()].T
                if v.dtype.names:v=v['real']+1j*v['imag']
                out[k]=v.reshape(g+(v.shape[-1],),order='F')
        return out
    def frame(self,index):
        f=super().frame(index)
        f.images['magnetic_delta']=f.images['magnetic_z']-self._initial_b
        return self.spin_fields(f,index)
    def spin_fields(self,f,index):
        psi=self.load_snapshot(index)['psi'][:,:,0,:].transpose(1,0,2);a=psi[...,1];b=psi[...,2]
        phase=np.angle(a.conj()*b);phase[abs(a*b)<max(float(abs(a*b).max())*1e-7,1e-30)]=np.nan
        f.images.update(real_2=a.real,imag_2=a.imag,real_3=b.real,imag_3=b.imag,relative_phase=phase,
          spin_z=self.physics['hbar']/2*np.sum(abs(psi)**2*np.array([1,-1,1,-1]),axis=-1))
        return f
