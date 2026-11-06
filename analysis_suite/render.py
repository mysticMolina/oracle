"""Streaming, fixed-scale movie and analysis renderer for both solvers.

Jacob Matthew Molina. Inception: 2026-10-08.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import imageio_ffmpeg
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import Normalize
import numpy as np

from .adapters import Run, sha256
from .presentation import time_axis, time_title, physical_name, field_norm

plt.rcParams.update({"font.size":13,"axes.titlesize":14,"axes.labelsize":13,"xtick.labelsize":11,"ytick.labelsize":11})


LABELS = {
    'spin_z': (r'Spin angular momentum density Sz', 'RdBu_r'),
    'density': (r'Probability density', 'magma'),
    'current': (r'Probability current on density', 'magma'),
    'psi_real': (r'Re $\psi$', 'RdBu_r'),
    'psi_imag': (r'Im $\psi$', 'RdBu_r'),
    'phase': (r'Phase of $\psi$', 'twilight'),
    'electric': (r'$|E_{xy}|$ and direction', 'viridis'),
    'electric_energy': (r'$|Y|^2/2$ (normalized)', 'viridis'),
    'magnetic_delta': (r'$\delta B_z$', 'RdBu_r'),
    'magnetic_total': (r'$B_z$ total', 'RdBu_r'),
    'magnetic_z': (r'$B_z$ from saved $A$', 'RdBu_r'),
    'poynting': (r'$|S_{xy}|$ and direction', 'inferno'),
    'em_energy': (r'Electromagnetic energy density', 'inferno'),
    'ham_quantum_gradient': (r'$H_{\rm Q,gradient}$ density', 'RdBu_r'),
    'ham_quantum_coupling': (r'$H_{\rm Q,coupling}$ density', 'RdBu_r'),
    'ham_quantum_potential': (r'$H_{\rm Q,potential}$ density', 'RdBu_r'),
    'ham_electric': (r'$H_{\rm E}$ density', 'viridis'),
    'ham_magnetic': (r'$H_{\rm B}$ density', 'viridis'),
    'ham_total': (r'$H_{\rm total}$ density', 'RdBu_r'),
    'ham_total_excess': (r'$\mathcal{H}-\mathcal{H}_{B_0}$', 'RdBu_r'),
    'ham_dirac_rest': (r'Dirac rest energy density', 'RdBu_r'),
    'ham_dirac_kinetic': (r'Dirac kinetic energy density', 'RdBu_r'),
    'ham_dirac_gauge': (r'Dirac gauge coupling density', 'RdBu_r'),
    'ham_dirac_total': (r'Dirac matter energy density', 'RdBu_r'),
    'component_2': (r'Upper spin-down density ($|\psi_2|^2$)', 'magma'),
    'component_3': (r'Relativistic partner density ($|\psi_3|^2$)', 'magma'),
    'real_2': (r'Upper spin-down: Re $\psi_2$', 'RdBu_r'),
    'imag_2': (r'Upper spin-down: Im $\psi_2$', 'RdBu_r'),
    'real_3': (r'Lower spin-up: Re $\psi_3$', 'RdBu_r'),
    'imag_3': (r'Lower spin-up: Im $\psi_3$', 'RdBu_r'),
    'relative_phase': (r'Partner relative phase: arg($\psi_2^*\psi_3$)', 'twilight'),
}

# 2022 CODATA, NIST SRD 121. The simulation parameters remain in atomic units.
ATOMIC_SI = {'a0_m':5.29177210544e-11,'t0_s':2.4188843265864e-17,
             'E0_V_per_m':5.14220675112e11,'B0_T':2.35051757077e5,
             'Eh_J':4.3597447222060e-18}
_DENSITY_CONTOUR_FRACTIONS = np.asarray([.02,.08,.16,.24])/.314
_MAGNETIC_CONTOUR_FRACTIONS = np.asarray([-1/6,-1/12,0,1/12,1/6])
_DENSITY_FIELDS = {'density','current','component_2','component_3'}
_MAGNETIC_FIELDS = {'magnetic_delta','magnetic_z'}


def _coords(run: Run) -> tuple[np.ndarray,np.ndarray]:
    return np.asarray(run.plot_x),np.asarray(run.plot_y)


def _field_scale(run: Run, field: str) -> float:
    if run.kind != 'sphinx':
        return 1.0
    if field in ('density','current'):
        return 0.5*run.psi0**2
    if field in ('psi_real','psi_imag'):
        return run.psi0/np.sqrt(2)
    if field == 'electric':
        return run.energy0/run.lambda0
    if field == 'electric_energy':
        return run.y0**2/float(run.physics['eps_0'])
    if field in ('magnetic_delta','magnetic_total'):
        return run.bscale
    if field == 'poynting':
        return run.y0*run.bscale/(float(run.physics['eps_0'])*run.mu)
    return 1.0


def _vector_scale(run: Run, field: str) -> float:
    if run.kind != 'sphinx':
        return 1.0
    if field == 'current':
        return run.y0/(float(run.physics['hbar'])/run.energy0)
    return _field_scale(run,field)


def _label(run: Run, field: str) -> str:
    title = LABELS[field][0]
    if getattr(run,'meta',{}).get('solverVariant') in ('oracle_matlab_planar_v1','oracle_matlab_planar_v2','oracle_matlab_planar_v3') and field=='electric':
        title=r'$|\mathbf{E}|$; arrows: $xy$ projection'
    if getattr(run,'meta',{}).get('solverVariant') in ('sphinx_matched_forward_maxwell','oracle_matlab_planar_v1','oracle_matlab_planar_v2','oracle_matlab_planar_v3'):
        if field=='ham_magnetic':title=r'Perturbation magnetic energy density'
        if field=='magnetic_z':title=r'$B_0+(\nabla\times\delta A)_z$'
    if run.kind=='oracle_static':return title
    if field in _DENSITY_FIELDS:
        power = 2 if run.kind == 'oracle_static' else 3
        return title+rf' ($a_0^{{-{power}}}$)'
    if field in ('psi_real','psi_imag'):
        return title+r' ($a_0^{-3/2}$)'
    if field in ('real_2','imag_2','real_3','imag_3'):
        return title+r' ($a_0^{-3/2}$)'
    if field == 'electric':
        return title+r' ($E_{\rm au}$)'
    if field in ('magnetic_delta','magnetic_total','magnetic_z'):
        return title+r' ($B_{\rm au}$)'
    if field == 'poynting':
        return title+r' ($E_h/a_0^2t_{\rm au}$)'
    if field.startswith('ham_') or field in ('em_energy','electric_energy'):
        power = 2 if run.kind == 'oracle_static' else 3
        return title+rf' ($E_h/a_0^{power}$)'
    return title


def _add_ring(ax,run: Run) -> None:
    radius = getattr(run,'orbit_radius_a0',None)
    if radius is None or radius <= 0:
        return
    theta = np.linspace(0,2*np.pi,512)
    line, = ax.plot(radius*np.cos(theta),radius*np.sin(theta),'--',
                    color='white',lw=1.7,alpha=.95,zorder=5)
    line.set_path_effects([path_effects.Stroke(linewidth=3.2,foreground='#111827'),
                           path_effects.Normal()])


def _contour(ax,run: Run,field: str,image: np.ndarray,peak: float):
    if field in _DENSITY_FIELDS:
        levels = _DENSITY_CONTOUR_FRACTIONS*peak
    elif field in _MAGNETIC_FIELDS:
        levels = _MAGNETIC_CONTOUR_FRACTIONS*peak
    else:
        return None
    low,high = float(np.nanmin(image)),float(np.nanmax(image))
    levels = levels[(levels > low) & (levels < high)]
    if not len(levels):
        return None
    x,y = _coords(run)
    return ax.contour(x,y,image,levels=levels,colors='white',
                      linewidths=1.15,alpha=.85,zorder=4)


def preset(run: Run, requested: str = 'auto') -> list[str]:
    if requested == 'auto':
        if run.kind == 'sphinx':
            condition = str(run.problem['initialCondition'])
            requested = 'landau' if condition.startswith('paper_landau') else 'coupled_cyclotron'
        elif run.kind == 'oracle_static':
            requested = 'oracle_static'
        else:
            requested = 'oracle_coupled'
    choices = {
        'cyclotron': ['density','current','electric','magnetic_delta','psi_real','psi_imag'],
        'landau': ['density','current','psi_real','psi_imag','electric','magnetic_delta'],
        'coupled_cyclotron': ['density','current','electric','magnetic_delta','poynting','ham_total_excess'],
        'oracle_static': ['density','current','component_2','component_3','ham_dirac_rest','ham_dirac_kinetic'],
        'oracle_coupled': ['density','current','electric','magnetic_delta','poynting','ham_total'],
    }
    if requested not in choices:
        raise ValueError(f'Unknown preset {requested}')
    fields = [field for field in choices[requested] if field in run.available]
    if not fields:
        raise ValueError('No available fields for this preset and run')
    return fields


def preflight(run: Run, fields: list[str]) -> dict:
    if not np.all(np.diff(run.times) > 0):
        raise ValueError('Saved times must increase strictly')
    peaks = {name: 0.0 for name in fields}
    vector_peaks = {name: 0.0 for name in fields}
    integrals = []
    centroid_x = []
    centroid_y = []
    hamiltonian = {name: [] for name in fields if name.startswith('ham_')}
    plot_x,plot_y = _coords(run)
    dx = float(plot_x[1]-plot_x[0]); dy = float(plot_y[1]-plot_y[0])
    def integrate(image: np.ndarray) -> float:
        if run.kind == 'sphinx':
            # NumPy 2 removed trapz; NumPy 1 installations may lack trapezoid.
            trapezoid = getattr(np, 'trapezoid', None)
            if trapezoid is None:
                trapezoid = np.trapz
            return float(trapezoid(trapezoid(image,plot_x,axis=1),plot_y,axis=0))
        return float(np.sum(image)*dx*dy)
    for index in range(len(run.times)):
        frame = run.frame(index)
        shape = (len(run.y),len(run.x))
        for name in fields:
            image = np.asarray(frame.images[name])
            phase = name in ('phase', 'relative_phase')
            if image.shape != shape or np.isinf(image).any() or (not phase and not np.isfinite(image).all()):
                raise ValueError(f'Invalid {name} frame at index {index}')
            if not np.isfinite(image).any():
                raise ValueError(f'Empty {name} frame at index {index}')
            peaks[name] = max(peaks[name],float(np.nanmax(abs(image))))
            if name in frame.vectors:
                vx, vy = frame.vectors[name]
                if vx.shape != shape or vy.shape != shape or not np.isfinite(vx).all() or not np.isfinite(vy).all():
                    raise ValueError(f'Invalid {name} vectors at index {index}')
                vector_peaks[name] = max(vector_peaks[name],float(np.max(np.hypot(vx,vy))))
        density = frame.images['density']*_field_scale(run,'density')
        integral = integrate(density)
        if not np.isfinite(integral) or integral <= 0:
            raise ValueError(f'Invalid probability integral at index {index}')
        integrals.append(integral)
        centroid_x.append(integrate(density*plot_x[None,:])/integral)
        centroid_y.append(integrate(density*plot_y[:,None])/integral)
        for name in hamiltonian:
            hamiltonian[name].append(integrate(frame.images[name]*_field_scale(run,name)))
    return {'peaks':peaks,'vectorPeaks':vector_peaks,'planarIntegral':integrals,
            'centroidX':centroid_x,'centroidY':centroid_y,
            'hamiltonianAreaIntegrals':hamiltonian,
            'planarIntegralDrift':float(max(abs(np.asarray(integrals)-integrals[0]))/integrals[0])}


def _bounds(field: str, peak: float) -> tuple[float,float]:
    scale = peak if peak > 0 else 1.0
    if field in ('phase','relative_phase'):
        return -np.pi,np.pi
    if field in ('psi_real','psi_imag','real_2','imag_2','real_3','imag_3',
                 'magnetic_delta','magnetic_z','magnetic_total',
                 'ham_quantum_gradient','ham_quantum_coupling',
                 'ham_quantum_potential','ham_total','ham_total_excess',
                 'ham_dirac_rest','ham_dirac_kinetic','ham_dirac_gauge','ham_dirac_total','spin_z'):
        return -scale,scale
    return 0,scale


def integrated_hamiltonian_lineouts(run: Run, output: Path, audit: dict) -> dict | None:
    """Paper-style, full-domain Hamiltonian traces for corrected SPHINX runs."""
    h = audit['hamiltonianAreaIntegrals']
    needed = ('ham_quantum_gradient','ham_quantum_coupling','ham_quantum_potential',
              'ham_electric','ham_magnetic','ham_total')
    if run.kind != 'sphinx' or any(name not in h for name in needed):
        return None
    gradient,coupling,potential,electric,magnetic,total = (
        np.asarray(h[name],dtype=float) for name in needed)
    quantum = gradient+coupling+potential
    electromagnetic = electric+magnetic
    probability = np.asarray(audit['planarIntegral'],dtype=float)
    h0,p0 = float(total[0]),float(probability[0])
    if abs(h0) <= 1e-300 or p0 <= 0:
        raise ValueError('Cannot normalize SPHINX integrated Hamiltonian lineouts')
    omega = abs(float(run.physics['q'])*float(run.b0)/float(run.physics['m']))
    tc = 2*np.pi/omega if omega > 0 else None
    time = np.asarray(run.times,dtype=float)
    horizontal = time/tc if tc else time
    xlabel = r'$t/T_c$' if tc else 't (a.u.)'
    em_fraction = np.divide(np.column_stack((magnetic,electric)),electromagnetic[:,None],
                            out=np.full((len(time),2),np.nan),
                            where=abs(electromagnetic[:,None])>1e-300)
    qm_fraction = np.divide(np.column_stack((gradient,coupling,potential)),quantum[:,None],
                            out=np.full((len(time),3),np.nan),
                            where=abs(quantum[:,None])>1e-300)
    traces = [
        (r'$H_{\rm em}/H_0$',[(electromagnetic/h0,'EM','#2874c6')]),
        (r'$H_{\rm qm}/H_0$',[(quantum/h0,'QM','#e05252')]),
        (r'$H_{\rm total}/H_0$',[(total/h0,'total','#35a85b')]),
        (r'$P/P_0$',[(probability/p0,'probability','#2874c6')]),
        ('EM energy partition',[(em_fraction[:,0],'magnetic','#2874c6'),
                                (em_fraction[:,1],'electric','#e05252')]),
        ('QM energy partition',[(qm_fraction[:,0],'gradient','#2874c6'),
                                (qm_fraction[:,1],'coupling','#e05252'),
                                (qm_fraction[:,2],'potential','#35a85b')]),
        (r'$(H_{\rm total}-H_0)/H_0$',[((total-h0)/h0,'energy drift','#35a85b')]),
        (r'$(P-P_0)/P_0$',[((probability-p0)/p0,'probability drift','#2874c6')]),
    ]
    fig,axes=plt.subplots(2,4,figsize=(19,8),layout='constrained',sharex=True)
    for index,(ax,(title,curves)) in enumerate(zip(axes.flat,traces)):
        for values,label,color in curves:
            ax.plot(horizontal,values,lw=2,color=color,label=label)
        ax.set_title(title,fontsize=12)
        ax.set_xlabel(xlabel)
        ax.grid(alpha=.25)
        if index in (6,7):
            ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0),useOffset=False)
        else:
            ax.ticklabel_format(axis='y',style='plain',useOffset=False)
        if index in (2,3):
            ax.set_ylim(0,1.05*max(1.0,max(float(np.nanmax(v)) for v,_,_ in curves)))
        if len(curves)>1:ax.legend(fontsize=8,frameon=False)
        if index in (6,7):
            finite=np.concatenate([v[np.isfinite(v)] for v,_,_ in curves])
            span=max(float(np.max(abs(finite))) if len(finite) else 0,1e-12)
            ax.set_ylim(-1.1*span,1.1*span)
    fig.suptitle(f'{run.name[:32]} | integrated Hamiltonian and probability',fontsize=16)
    figure=output/'figures'/'hamiltonian_integral_lineouts.png'
    fig.savefig(figure,dpi=140,bbox_inches='tight')
    plt.close(fig)
    csvpath=output/'hamiltonian_integral_lineouts.csv'
    with csvpath.open('w',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['step','time_au','time_over_Tc','H_em_Eh_per_a0','H_qm_Eh_per_a0',
                         'H_total_Eh_per_a0','P_planar','H_magnetic_Eh_per_a0',
                         'H_electric_Eh_per_a0','H_gradient_Eh_per_a0',
                         'H_coupling_Eh_per_a0','H_potential_Eh_per_a0',
                         'H_em_over_H0','H_qm_over_H0','H_total_over_H0','P_over_P0',
                         'energy_relative_drift','probability_relative_drift'])
        for i in range(len(time)):
            writer.writerow([run.steps[i],time[i],horizontal[i] if tc else '',
                             electromagnetic[i],quantum[i],total[i],probability[i],
                             magnetic[i],electric[i],gradient[i],coupling[i],potential[i],
                             electromagnetic[i]/h0,quantum[i]/h0,total[i]/h0,
                             probability[i]/p0,(total[i]-h0)/h0,(probability[i]-p0)/p0])
    return {'figure':str(figure.relative_to(output)),'csv':str(csvpath.relative_to(output)),
            'H0_EhPerA0':h0,'P0Planar':p0,'cyclotronPeriodAtomicTime':tc,
            'spatialIntegral':'full 2D domain; energy per a0 without assumed z thickness'}


def evolution_figures(run: Run, output: Path, audit: dict, indices: np.ndarray) -> dict | None:
    folder = output/'figures'
    folder.mkdir(exist_ok=True)
    selected = [(int(i),run.frame(int(i))) for i in indices]
    plot_x,plot_y = _coords(run)
    extent = [float(plot_x[0]),float(plot_x[-1]),float(plot_y[0]),float(plot_y[-1])]
    columns = min(4,len(selected))
    rows = math.ceil(len(selected)/columns)
    for field in run.available:
        fig,axes = plt.subplots(rows,columns,figsize=(4.1*columns,3.7*rows),
                                squeeze=False,layout='constrained')
        label,cmap = LABELS[field]
        display_peak = audit['peaks'][field]*_field_scale(run,field)
        low,high = _bounds(field,display_peak)
        last_image = None
        for slot,(index,frame) in enumerate(selected):
            ax = axes.flat[slot]
            display_image = frame.images[field]*_field_scale(run,field)
            last_image = ax.imshow(display_image,origin='lower',extent=extent,
                                   cmap=cmap,norm=field_norm(field,low,high),interpolation='nearest')
            ax.set_title(time_title(run,index),fontsize=15)
            if run.kind=='sphinx':ax.set(xlim=(-3,3),ylim=(-3,3))
            ax.set_aspect('equal')
            ax.set_xlabel(r'$x$ (normalized)' if run.kind=='oracle_static' else r'$x/a_0$')
            ax.set_ylabel(r'$y$ (normalized)' if run.kind=='oracle_static' else r'$y/a_0$')
            if field in _DENSITY_FIELDS:
                _add_ring(ax,run)
            _contour(ax,run,field,display_image,display_peak)
            if field in frame.vectors and audit['vectorPeaks'][field] > 0:
                skip = max(1,min(len(run.x),len(run.y))//16)
                xx,yy = np.meshgrid(plot_x[::skip],plot_y[::skip])
                vx,vy = frame.vectors[field]
                spacing = min(float(plot_x[1]-plot_x[0]),float(plot_y[1]-plot_y[0]))
                scale = .8*skip*spacing/audit['vectorPeaks'][field]
                ax.quiver(xx,yy,vx[::skip,::skip]*scale,vy[::skip,::skip]*scale,
                          color='#00e5ff',pivot='mid',angles='xy',scale_units='xy',scale=1,width=.005)
        for ax in list(axes.flat)[len(selected):]:
            ax.axis('off')
        ring_note = (f' | orbit {run.orbit_radius_a0:.3g} a0'
                     if field in _DENSITY_FIELDS and getattr(run,'orbit_radius_a0',None) else '')
        fig.suptitle(physical_name(run)+' • '+LABELS[field][0],fontsize=18)
        fig.colorbar(last_image,ax=axes.ravel().tolist(),shrink=.6,pad=.01,label=_label(run,field)+(' (symlog)' if field in ('magnetic_delta','ham_total_excess') else ''))
        fig.savefig(folder/f'{field}_evolution.png',dpi=115)
        plt.close(fig)
    h = audit['hamiltonianAreaIntegrals']
    if h:
        fig,axes = plt.subplots(2,1,figsize=(12,8),sharex=True,layout='constrained')
        first = ('ham_quantum_gradient','ham_quantum_coupling','ham_quantum_potential') if run.kind == 'sphinx' else ('ham_dirac_rest','ham_dirac_kinetic','ham_dirac_gauge')
        second = ('ham_electric','ham_magnetic','ham_total') if run.kind != 'oracle_static' else ('ham_dirac_rest','ham_dirac_kinetic','ham_dirac_total')
        for key in first:
            if key in h:
                axes[0].plot(time_axis(run)[0],h[key],marker='o',markersize=3,label=LABELS[key][0])
        for key in second:
            if key in h:
                trace = np.asarray(h[key])
                axes[1].plot(time_axis(run)[0],trace-trace[0],marker='o',markersize=3,label=LABELS[key][0])
        for ax in axes:
            ax.grid(alpha=.25);ax.legend(fontsize=9)
        integral_unit = 'normalized' if run.kind == 'oracle_static' else r'$E_h/a_0$'
        axes[0].set_ylabel(f'2D area integral ({integral_unit})')
        axes[1].set_ylabel(f'Change in 2D area integral ({integral_unit})')
        axes[1].set_xlabel(time_axis(run)[1])
        fig.suptitle(physical_name(run)+' • Hamiltonian energetics',fontsize=18)
        fig.savefig(folder/'hamiltonian_subsystems.png',dpi=130)
        plt.close(fig)
    return integrated_hamiltonian_lineouts(run,output,audit)


class Canvas:
    def __init__(self, run: Run, fields: list[str], audit: dict):
        self.run, self.fields, self.audit = run, fields, audit
        self.fig = plt.figure(figsize=(16,12.5),dpi=100,facecolor='white')
        rows = 2 if len(fields) > 3 else 1
        columns = 3 if len(fields) > 3 else len(fields)
        grid = self.fig.add_gridspec(rows+1,columns,height_ratios=[1]*rows+[0.28],
                                    left=.05,right=.96,top=.89,bottom=.08,wspace=.32,hspace=.42)
        self.images = []
        self.axes = {}
        self.contours = {}
        self.quivers = []
        plot_x,plot_y = _coords(run)
        extent = [float(plot_x[0]),float(plot_x[-1]),float(plot_y[0]),float(plot_y[-1])]
        for slot, field in enumerate(fields):
            ax = self.fig.add_subplot(grid[slot//columns,slot%columns])
            _,cmap = LABELS[field]
            label = _label(run,field)
            peak = audit['peaks'][field]*_field_scale(run,field)
            bounds = _bounds(field,peak)
            image = ax.imshow(np.zeros((len(run.y),len(run.x))),origin='lower',
                              extent=extent,cmap=cmap,norm=field_norm(field,*bounds),interpolation='nearest')
            ax.set_title(label,fontsize=13,weight='bold',pad=6)
            ax.set_xlabel(r'$x$ (normalized)' if run.kind=='oracle_static' else r'$x/a_0$');ax.set_ylabel(r'$y$ (normalized)' if run.kind=='oracle_static' else r'$y/a_0$')
            ax.set_aspect('equal')
            cb=self.fig.colorbar(image,ax=ax,fraction=.046,pad=.025)
            if field in ('magnetic_delta','ham_total_excess'):
                cb.ax.set_title('symlog',fontsize=10,pad=5)
                ticks=np.asarray([-1,-.1,0,.1,1])*peak
                cb.set_ticks(ticks,labels=[f'{v:.2g}' for v in ticks])
            if run.kind=='sphinx':ax.set(xlim=(-3,3),ylim=(-3,3))
            self.images.append((field,image))
            self.axes[field] = ax
            if field in _DENSITY_FIELDS:
                _add_ring(ax,run)
            if field in ('current','electric','poynting'):
                skip = max(1,min(len(run.x),len(run.y))//16)
                xx, yy = np.meshgrid(plot_x[::skip],plot_y[::skip])
                q = ax.quiver(xx,yy,np.zeros_like(xx),np.zeros_like(yy),color='#00e5ff',
                              pivot='mid',angles='xy',scale_units='xy',scale=1,width=.005)
                field_peak = audit['vectorPeaks'][field]
                spacing = min(float(plot_x[1]-plot_x[0]),float(plot_y[1]-plot_y[0]))
                if field_peak > 0:
                    physical_peak = field_peak*_vector_scale(run,field)
                    ax.quiverkey(q,.64,.93,.8*skip*spacing,f'peak {physical_peak:.2g}',
                                 coordinates='axes',labelpos='E',labelcolor='#00e5ff',
                                 fontproperties={'size':8})
                self.quivers.append((field,q,skip))
        self.trace_ax = self.fig.add_subplot(grid[rows,:])
        x = np.asarray(run.times,dtype=float)
        norm = np.asarray(audit['planarIntegral'])
        self.trace_ax.plot(x,(norm-norm[0])/norm[0],color='#334155',label='Planar integral drift')
        if 'norm' in run.diagnostics and 'time' in run.diagnostics:
            d = run.diagnostics
            n = np.asarray(d['norm']);t = np.asarray(d['time'])
            self.trace_ax.plot(t,(n-n[0])/n[0],color='#7c3aed',alpha=.75,label='Stored full-domain norm drift')
        if 'gaussRelativeResidual' in run.diagnostics:
            g = run.diagnostics['gaussRelativeResidual']
            if len(g) > int(run.steps[-1]):
                self.trace_ax.plot(x,g[run.steps],color='#dc2626',label='Interior Gauss relative residual')
        self.trace_ax.set_xlabel('t (a.u.)');self.trace_ax.grid(alpha=.25)
        self.trace_ax.legend(loc='upper left',fontsize=8,ncol=2)
        self.marker = self.trace_ax.axvline(x[0],color='#f59e0b',lw=1.5)
        self.title = self.fig.suptitle('',fontsize=17,weight='bold',y=.97)
        ring_note = (f' | dashed orbit: {run.orbit_radius_a0:.3g} a0'
                     if getattr(run,'orbit_radius_a0',None) else '')
        self.footer = self.fig.text(.5,.015,
                                    r'$a_0=5.29177210544\times10^{-11}$ m; '
                                    r'$t_{\rm au}=2.4188843265864\times10^{-17}$ s'
                                    ' | fixed run-wide scales'+ring_note,
                                    ha='center',fontsize=9,color='#475569')

    def update(self,index: int):
        frame = self.run.frame(index)
        for field,image in self.images:
            display_image = frame.images[field]*_field_scale(self.run,field)
            image.set_data(display_image)
            previous = self.contours.pop(field,None)
            if previous is not None:
                if hasattr(previous, 'remove'):
                    previous.remove()
                else:
                    for artist in previous.collections:
                        artist.remove()
            current = _contour(self.axes[field],self.run,field,display_image,
                               self.audit['peaks'][field]*_field_scale(self.run,field))
            if current is not None:
                self.contours[field] = current
        for field,quiver,skip in self.quivers:
            vx,vy = frame.vectors.get(field,(None,None))
            if vx is None:
                continue
            peak = self.audit['vectorPeaks'][field]
            plot_x,plot_y = _coords(self.run)
            spacing = min(float(plot_x[1]-plot_x[0]),float(plot_y[1]-plot_y[0]))
            scale = .8*skip*spacing/max(peak,1e-300)
            quiver.set_UVC(vx[::skip,::skip]*scale,vy[::skip,::skip]*scale)
        self.marker.set_xdata([time_axis(self.run)[0][index]]*2)
        self.title.set_text(physical_name(self.run)+' • '+time_title(self.run,index))
        self.fig.canvas.draw()

    def rgb(self) -> bytes:
        rgba = np.asarray(self.fig.canvas.buffer_rgba())
        return np.ascontiguousarray(rgba[:,:,:3]).tobytes()

    def close(self):
        plt.close(self.fig)


class EnergyCanvas(Canvas):
    """Full movie with total-energy error and the solver's energy partition.

    Jacob Matthew Molina. Inception: 2026-10-12. All terms are the same
    saved-slice/planar area integrals used by the audit, divided by its first
    total H0. In particular, coupled ORACLE's slice is not its full 3D energy.
    """

    def __init__(self, run: Run, fields: list[str], audit: dict):
        super().__init__(run, fields, audit)
        self.trace_ax.remove()
        self.footer.remove()
        if len(fields) == 1:
            field_axis = self.axes[fields[0]]
            field_axis.set_position([.315, .40, .37, .48])
            for axis in list(self.fig.axes):
                if axis is not field_axis:
                    axis.set_position([.705, .40, .025, .48])
        elif len(fields) == 6:
            # Jacob Matthew Molina, 2026-10-12: Keep physical x/y scales square
            # while giving each saved field most of a 3-by-2 movie frame. The
            # colorbars travel with their image axes, so no data or labels are
            # hidden when the Hamiltonian plots occupy the bottom strip.
            field_axes = [self.axes[field] for field in fields]
            colorbars = [ax for ax in self.fig.axes if ax not in field_axes]
            if len(colorbars) != len(field_axes):
                raise ValueError('Expected one colorbar for each movie field')
            width, height = .218, .28
            for slot, (field_axis, colorbar) in enumerate(zip(field_axes, colorbars)):
                column, row = slot % 3, slot // 3
                x0 = .055 + .325 * column
                y0 = .63 if row == 0 else .285
                field_axis.set_position([x0, y0, width, height])
                colorbar.set_position([x0 + width + .008, y0 + .012, .012, height - .024])
        self.err_ax = self.fig.add_axes([.06, .08, .365, .13])
        self.energy_ax = self.fig.add_axes([.49, .08, .46, .13])
        time, time_label = time_axis(run)
        h = {key: np.asarray(values, dtype=float)
             for key, values in audit['hamiltonianAreaIntegrals'].items()}
        if run.kind == 'sphinx':
            total = h['ham_total']
            quantum = h['ham_quantum_gradient'] + h['ham_quantum_coupling'] + h['ham_quantum_potential']
            electromagnetic = h['ham_electric'] + h['ham_magnetic']
            curves = [
                (r'$H/H_0$', total, '#17131A', '-'),
                (r'$H_{\rm qm}/H_0$', quantum, '#5632B8', '-'),
                (r'$H_{\rm em}/H_0$', electromagnetic, '#168CA8', '-'),
                (r'$H_{\nabla}/H_0$', h['ham_quantum_gradient'], '#8B5CF6', '--'),
                (r'$H_{\rm coupling}/H_0$', h['ham_quantum_coupling'], '#C1378F', '--'),
                (r'$H_{\rm potential}/H_0$', h['ham_quantum_potential'], '#B06A08', '--'),
                (r'$H_E/H_0$', h['ham_electric'], '#0E7490', ':'),
                (r'$H_B/H_0$', h['ham_magnetic'], '#3B82F6', ':'),
            ]
            title = r'QM $\leftrightarrow$ EM and split energies / $H_0$'
            unit_note = r'full 2D integral; $E_h/a_0$'
        elif run.kind == 'oracle_static':
            total = h['ham_dirac_total']
            curves = [
                (r'$H_D/H_0$', total, '#17131A', '-'),
                (r'$H_{\rm rest}/H_0$', h['ham_dirac_rest'], '#5632B8', '-'),
                (r'$H_{\rm kinetic}/H_0$', h['ham_dirac_kinetic'], '#168CA8', '-'),
            ]
            title = r'Prescribed-field Dirac energy / $H_0$'
            unit_note = r'2D Dirac energy; $\hbar=m=c=|q|=1$'
        else:
            total = h['ham_total']
            quantum = h['ham_dirac_rest'] + h['ham_dirac_kinetic'] + h['ham_dirac_gauge']
            electromagnetic = h['ham_electric'] + h['ham_magnetic']
            curves = [
                (r'$H/H_0$', total, '#17131A', '-'),
                (r'$H_D/H_0$', quantum, '#5632B8', '-'),
                (r'$H_{\rm em}/H_0$', electromagnetic, '#168CA8', '-'),
                (r'$H_{\rm rest}/H_0$', h['ham_dirac_rest'], '#8B5CF6', '--'),
                (r'$H_{\rm kinetic}/H_0$', h['ham_dirac_kinetic'], '#C1378F', '--'),
                (r'$H_{\rm gauge}/H_0$', h['ham_dirac_gauge'], '#B06A08', '--'),
                (r'$H_E/H_0$', h['ham_electric'], '#0E7490', ':'),
                (r'$H_B/H_0$', h['ham_magnetic'], '#3B82F6', ':'),
            ]
            title = r'Dirac $\leftrightarrow$ EM energies / $H_0$'
            unit_note = r'full 2D area integral; per unit z length' if run.kind=='oracle_longshot' else r'central $z$ slice area integral, not full 3D energy'
        h0 = float(total[0])
        if not np.isfinite(h0) or abs(h0) < 1e-12:
            raise ValueError('A finite nonzero initial total Hamiltonian is required')
        self.err_ax.plot(time, (total-h0)/h0, color='#5632B8', lw=1.8)
        self.err_ax.axhline(0, color='#666', lw=.65)
        self.err_ax.set_title(r'Total-energy error $(H-H_0)/H_0$', fontsize=13, pad=7)
        if 'cumulativeProjectionEnergy' in getattr(run,'saved',{}):
            saved=run.saved;ts=saved['time'];projection=saved['cumulativeProjectionEnergy']
            corrected=saved['totalEnergy']-projection-saved['totalEnergy'][0]
            self.err_ax.lines[0].set_label('raw drift')
            self.err_ax.plot(ts,corrected/h0,color='#168CA8',lw=1.3,label='minus projection work')
            self.err_ax.plot(ts,projection/h0,color='#B06A08',ls=':',lw=1.1,label='projection work')
            self.err_ax.set_title('Energy drift / H0; projection work separated',fontsize=12,pad=7)
            self.err_ax.legend(fontsize=8,loc='best')
            if 'cumulativeBoundaryWork' in saved:
                boundary=saved['cumulativeBoundaryWork']
                balance=corrected-boundary
                self.err_ax.plot(ts,balance/h0,color='#28612C',lw=1.4,label='minus projection & boundary work')
                self.err_ax.plot(ts,boundary/h0,color='#A24C6C',ls='--',lw=1,label='boundary work')
                self.err_ax.legend(fontsize=8,loc='best')
        self.err_ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useOffset=False)
        self.err_ax.yaxis.get_offset_text().set_fontsize(11)
        for label, values, color, style in curves:
            self.energy_ax.plot(time, values/h0, color=color, ls=style,
                                lw=1.65 if style == '-' else 1.1, label=label)
        self.energy_ax.set_title(title, fontsize=13, pad=7)
        self.energy_ax.legend(loc='upper center', bbox_to_anchor=(.5, -.30),
                              fontsize=10.5, ncol=4 if len(curves) > 3 else 3,
                              framealpha=.94, facecolor='white', edgecolor='#ddd',
                              handlelength=1.5, columnspacing=.6, labelspacing=.15,
                              borderpad=.25)
        for axis in (self.err_ax, self.energy_ax):
            axis.grid(alpha=.2)
            axis.tick_params(labelsize=11, length=3, pad=2)
            axis.set_xlabel(time_label, fontsize=12, labelpad=1)
            axis.set_xlim(float(time[0]), float(time[-1]))
            axis.spines[['top', 'right']].set_visible(False)
        self.marker = self.err_ax.axvline(time[0], color='#f59e0b', lw=1.3)
        self.energy_marker = self.energy_ax.axvline(time[0], color='#f59e0b', lw=1.3)
        self.fig.text(.24, .012, rf'$H_0={h0:.5g}$; {unit_note}',
                      ha='center', fontsize=10, color='black')

    def update(self, index: int):
        self.energy_marker.set_xdata([time_axis(self.run)[0][index]]*2)
        super().update(index)


def _decode(path: Path) -> dict:
    reader = imageio_ffmpeg.read_frames(str(path))
    meta = next(reader)
    count = sum(1 for _ in reader)
    return {'frames':count,'fps':meta['fps'],'size':list(meta['size']),
            'codec':meta['codec'],'sha256':sha256(path),'bytes':path.stat().st_size}


def render(run: Run, output: Path, *, preset_name: str = 'auto', fps: float = 8,
           stride: int | None = None, target_frames: int = 80,
           snapshots: int = 12, preview_only: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {output}')
    if run.source == output or run.source in output.parents:
        raise ValueError('Output must be outside the raw simulation folder')
    if (stride is not None and stride < 1) or fps <= 0 or target_frames < 1 or snapshots < 1:
        raise ValueError('stride, target frames, snapshots, and fps must be positive')
    fields = preset(run,preset_name)
    output.mkdir(parents=True,exist_ok=True)
    marker = output / 'INCOMPLETE'
    marker.write_text('Analysis in progress or failed.\n')
    audit = preflight(run,list(run.available))
    if stride is None:
        stride = max(1,math.ceil((len(run.times)-1)/max(1,target_frames-1)))
    indices = np.unique(np.r_[np.arange(0,len(run.times),stride),len(run.times)-1]).astype(int)
    still_indices = np.unique(np.linspace(0,len(run.times)-1,min(snapshots,len(run.times)),dtype=int))
    energy_available = ('ham_dirac_total' in audit['hamiltonianAreaIntegrals']
                        if run.kind == 'oracle_static'
                        else 'ham_total' in audit['hamiltonianAreaIntegrals'])
    canvas = EnergyCanvas(run,fields,audit) if energy_available else Canvas(run,fields,audit)
    try:
        lineout_metadata = evolution_figures(run,output,audit,still_indices)
        still_folder = output/'snapshots'
        still_folder.mkdir(exist_ok=True)
        for index in still_indices:
            canvas.update(int(index))
            canvas.fig.savefig(still_folder/f'frame_{int(index):04d}.png',dpi=100)
        for name,index in [('first',0),('middle',len(run.times)//2),('last',len(run.times)-1)]:
            canvas.update(index)
            canvas.fig.savefig(output/f'{name}.png',dpi=100)
        movie = None
        if not preview_only:
            path = output/'movie.partial.mp4'
            writer = imageio_ffmpeg.write_frames(str(path),canvas.fig.canvas.get_width_height(),fps=fps,
                                                 codec='libx264',pix_fmt_in='rgb24',
                                                 pix_fmt_out='yuv420p',macro_block_size=2,
                                                 output_params=['-crf','18'])
            writer.send(None)
            try:
                for index in indices:
                    canvas.update(int(index))
                    writer.send(canvas.rgb())
            finally:
                writer.close()
            movie = _decode(path)
            if movie['frames'] != len(indices) or abs(movie['fps']-fps) > 1e-3:
                raise ValueError('Encoded movie frame count or fps mismatch')
            path.rename(output/'movie.mp4')
        with (output/'diagnostics.csv').open('w',newline='') as stream:
            csvwriter = csv.writer(stream)
            hnames = list(audit['hamiltonianAreaIntegrals'])
            csvwriter.writerow(['step','time','planarIntegral','centroidX','centroidY',*hnames])
            for i in range(len(run.times)):
                csvwriter.writerow([run.steps[i],run.times[i],audit['planarIntegral'][i],
                                    audit['centroidX'][i],audit['centroidY'][i],
                                    *[audit['hamiltonianAreaIntegrals'][name][i] for name in hnames]])
        receipt = dict(schemaVersion=1,solver=run.kind,source=str(run.source),name=run.name,
                       status=run.status,notes=run.notes,fields=fields,framesAvailable=len(run.times),
                       framesRendered=len(indices),frameIndices=indices.tolist(),
                       snapshots=still_indices.tolist(),fieldsAnalyzed=list(run.available),
                       firstTime=float(run.times[0]),lastTime=float(run.times[-1]),
                       fps=fps,stride=stride,audit=audit,movie=movie,
                       movieLayout='energy' if energy_available else 'legacy_no_energy_terms',
                       presentation={'lengthUnit':('normalized' if run.kind=='oracle_static' else 'a0'),'timeUnit':time_axis(run)[1],
                                     'orbitRadiusA0':getattr(run,'orbit_radius_a0',None),
                                     'densityContourFractions':_DENSITY_CONTOUR_FRACTIONS.tolist(),
                                     'magneticContourFractions':_MAGNETIC_CONTOUR_FRACTIONS.tolist(),
                                     'fieldScalesFromNative':{name:_field_scale(run,name) for name in run.available},
                                     'atomicSI':ATOMIC_SI},
                       integratedHamiltonianLineouts=lineout_metadata,
                       implementationSha256=sha256(Path(__file__)),
                       sourceFirstLastSha256={
                           'first':sha256((getattr(run,'files',None) or run.s_files)[0]),
                           'last':sha256((getattr(run,'files',None) or run.s_files)[-1]),
                       })
        if hasattr(run,'f_files'):
            receipt['fieldFirstLastSha256'] = {
                'first':sha256(run.f_files[0]), 'last':sha256(run.f_files[-1])}
        (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        marker.rename(output/'COMPLETE' if run.status == 'completed' else output/'PARTIAL')
        return receipt
    finally:
        canvas.close()
