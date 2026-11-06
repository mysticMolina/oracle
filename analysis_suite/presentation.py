"""Jacob Matthew Molina, 2026-10-13: physical labels shared by diagnostics."""
import numpy as np
from matplotlib.colors import Normalize, SymLogNorm

def time_axis(run):
    """Return displayed times and label; SPHINX saved times are in solver tau."""
    if run.kind=='sphinx':
        tau=float(run.physics['hbar'])/run.energy0
        omega=abs(float(run.physics['q'])*run.b0*run.bscale/float(run.physics['m']))
        if omega>0:return np.asarray(run.times)*tau*omega/(2*np.pi),r'$t/T_c$'
        return np.asarray(run.times)*tau,'t (a.u.)'
    return np.asarray(run.times),'t (normalized)' if run.kind=='oracle_static' else 't (a.u.)'

def time_title(run,index):
    t,label=time_axis(run)
    return f'{label} = {t[index]:.3g}'

def _physical_name(run):
    if run.kind=='sphinx':
        initial=str(run.problem['initialCondition'])
        level=run.problem.get('landau',{})
        if 'landau' in initial:
            import re
            match=re.search(r'n_(\d+)_m_(\d+)',run.name)
            state=(rf'Landau $n={match[1]},\ m={match[2]}$' if match else 'Landau state')
        else:state='Coherent packet'
        return rf'SPHINX • {state} • $\beta={run.b_background:g}$'
    if run.kind=='oracle_longshot':
        label=f'ORACLE Dirac | {run.name} | c={run.physics["c"]:.6g}'
        meta=getattr(run,'meta',{})
        if str(meta.get('solverVariant','')).startswith('oracle_matlab_planar'):
            boundary=meta.get('boundaryCondition','')
            label+=' | '+{'mur1':'Mur 1','mur2':'Mur 2 [UNVALIDATED]',
                         'fixed_corrected_PEC_tangential':'PEC'}.get(boundary,boundary)
        return label
    if run.kind=='oracle_static':return rf'ORACLE • Dirac packet • $B_0={float(run.p["B"]):g}$'
    return 'ORACLE • Coupled Dirac–Maxwell'

def field_norm(field,low,high):
    if field in ('magnetic_delta','ham_total_excess') and high>0:
        return SymLogNorm(linthresh=.02*high,linscale=.7,vmin=low,vmax=high,base=10)
    return Normalize(low,high)

def canonical_guide(run):
    """Ideal centroid in symmetric gauge A=B(-y,x)/2; mechanical p is twice canonical p.

    Applies only to the initialized centered cyclotron coherent orbit. The
    nonlinear coupled evolution need not stay on the prescribed-field guide.
    """
    r=getattr(run,'orbit_radius_a0',None)
    if run.kind!='sphinx' or not r:return None
    return r,abs(float(run.physics['q'])*run.b0*run.bscale)*r/2


def physical_name(run):
    name=_physical_name(run)
    if len(run.times)<=2:name+=" • 2 saved states"
    return name if run.status=="completed" else name+" • PARTIAL saved history"
