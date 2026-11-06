"""Jacob Matthew Molina, 2026-10-15. Matched-campaign Maxwell convention.

Dirac derivatives remain centered. Maxwell uses forward curl and -G^T
Gauss divergence; uniform imposed B is separate from perturbation A.
"""
import numpy as np
from .dirac_longshot import DiracLongshotRun
from .adapters import OracleCoupledRun,_forward

class MatchedDiracRun(DiracLongshotRun):
 def __init__(self,source,allow_partial=False):
  super().__init__(source,allow_partial)
  self.notes += ['Compatible forward Maxwell curl; imposed uniform B supplied separately.',
                 'Hamiltonian magnetic term uses perturbation B; imposed-field background and cross energy excluded.',
                 'Gauss projection energy is recorded in coupled.csv, not hidden as conservation.']
 def fields(self,index):
  s=self.load_snapshot(index);A=s['A'][:,:,0,:].transpose(1,0,2);Y=s['Y'][:,:,0,:].transpose(1,0,2)
  xx,yy=np.meshgrid(self.x,self.y);B=float(self.meta['B']);a0=np.stack([-B*yy/2,B*xx/2,np.zeros_like(xx)],axis=-1)
  da=A-a0;db=_forward(da[...,1],self.x[1]-self.x[0],1)-_forward(da[...,0],self.y[1]-self.y[0],0)
  return -Y/self.physics['eps0'],db
 def frame(self,index):
  f=OracleCoupledRun.frame(self,index);E,db=self.fields(index);eps=self.physics['eps0'];c=self.physics['c'];B=float(self.meta['B'])+db
  hm=eps*c*c*db*db/2;he=eps*np.sum(E*E,axis=-1)/2
  sx=eps*c*c*E[...,1]*B;sy=-eps*c*c*E[...,0]*B
  f.images.update(magnetic_z=B,magnetic_delta=db,ham_magnetic=hm,ham_electric=he,em_energy=he+hm,poynting=np.hypot(sx,sy))
  f.images['ham_total']=sum(f.images[k] for k in ('ham_dirac_rest','ham_dirac_kinetic','ham_dirac_gauge','ham_electric','ham_magnetic'))
  f.vectors['poynting']=(sx,sy)
  return self.spin_fields(f,index)
