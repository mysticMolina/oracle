"""Read-only adapters for SPHINX and ORACLE simulation folders.

Jacob Matthew Molina. Inception: 2026-10-08.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class Frame:
    images: dict[str, np.ndarray]
    vectors: dict[str, tuple[np.ndarray, np.ndarray]]


class Run:
    kind: str
    source: Path
    name: str
    status: str
    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    steps: np.ndarray
    available: tuple[str, ...]
    notes: list[str]
    diagnostics: dict[str, np.ndarray]

    def frame(self, index: int) -> Frame:
        raise NotImplementedError


def _files(folder: Path, prefix: str, suffix: str) -> tuple[np.ndarray, list[Path]]:
    pairs = []
    for path in folder.glob(f'{prefix}_*.{suffix}'):
        try:
            pairs.append((int(path.stem.split('_')[-1]), path))
        except ValueError:
            continue
    pairs.sort()
    if not pairs:
        raise ValueError(f'No {prefix} snapshots in {folder}')
    return np.asarray([step for step, _ in pairs]), [path for _, path in pairs]


def _forward(field: np.ndarray, h: float, axis: int) -> np.ndarray:
    result = np.zeros_like(field)
    current = [slice(None)] * field.ndim
    ahead = [slice(None)] * field.ndim
    current[axis] = slice(None, -1)
    ahead[axis] = slice(1, None)
    result[tuple(current)] = (field[tuple(ahead)] - field[tuple(current)]) / h
    return result


def _adjoint_forward(field: np.ndarray, h: float, axis: int) -> np.ndarray:
    result = np.zeros_like(field)
    current = [slice(None)] * field.ndim
    ahead = [slice(None)] * field.ndim
    current[axis] = slice(None, -1)
    ahead[axis] = slice(1, None)
    result[tuple(current)] -= field[tuple(current)] / h
    result[tuple(ahead)] += field[tuple(current)] / h
    return result


def _centered_zero(field: np.ndarray, h: float, axis: int, order: int = 2) -> np.ndarray:
    offsets = {-2:1, -1:-8, 1:8, 2:-1} if order == 4 else {-1:-1,1:1}
    denominator = 12*h if order == 4 else 2*h
    result = np.zeros_like(field)
    for shift,weight in offsets.items():
        target = [slice(None)]*field.ndim
        source = [slice(None)]*field.ndim
        target[axis] = slice(max(0,-shift), min(field.shape[axis],field.shape[axis]-shift))
        source[axis] = slice(max(0,shift), min(field.shape[axis],field.shape[axis]+shift))
        result[tuple(target)] += weight*field[tuple(source)]/denominator
    return result


def _centered_projected(field: np.ndarray, h: float, axis: int) -> np.ndarray:
    result = _centered_zero(field,h,axis)
    interior = tuple(slice(1,-1) if field.shape[k]>1 else slice(None) for k in range(3))
    projected = np.zeros_like(result)
    projected[interior] = result[interior]
    return projected


def _backward_periodic(field: np.ndarray, h: float, axis: int) -> np.ndarray:
    return (field - np.roll(field, 1, axis=axis)) / h


class SphinxRun(Run):
    def __init__(self, source: Path, allow_partial: bool = False):
        self.source = source.resolve()
        manifest = json.loads((source / 'manifest.json').read_text())
        self.problem = manifest['problem']
        grid = self.problem['domain']['grid']
        nx, ny, nz = map(int, grid)
        if nz != 1:
            raise ValueError('SPHINX movie adapter currently requires a 2D run')
        self.nx, self.ny, self.n = nx, ny, nx * ny
        self.kind = 'sphinx'
        self.name = str(self.problem.get('name', source.name))
        self.status = manifest.get('status', 'unknown')
        self.notes = []
        if self.status != 'completed' and not allow_partial:
            raise ValueError(f'{source}: {self.status}; pass --allow-partial to render saved frames')
        s_steps, self.s_files = _files(source / 'S', 'S', 'txt')
        f_steps, self.f_files = _files(source / 'F', 'F', 'txt')
        if not np.array_equal(s_steps, f_steps):
            raise ValueError('SPHINX S/F snapshot schedules differ')
        self.steps = s_steps
        expected = np.arange(0, int(self.problem['time']['steps']), int(self.problem['output']['every']))
        if self.status == 'completed' and not np.array_equal(self.steps, expected):
            raise ValueError('Completed SPHINX run has an incomplete snapshot schedule')
        if self.status != 'completed':
            self.notes.append('PARTIAL: only paired saved snapshots are rendered')
        limits = np.asarray(self.problem['domain']['extentLambda'], dtype=float)
        self.x = np.linspace(*limits[0], nx)
        self.y = np.linspace(*limits[1], ny)
        final = self.problem['time'].get('endTime')
        if final is None or final == []:
            raise ValueError('SPHINX analysis requires explicit endTime in manifest')
        self.times = self.steps * float(final) / (int(self.problem['time']['steps']) - 1)
        self.boundary = self.problem['boundary']
        self.physics = self.problem['physics']
        hbar = float(self.physics['hbar'])
        eps = float(self.physics['eps_0'])
        lam = 4 * np.pi * eps * hbar**2
        energy = (hbar / lam)**2
        tau = hbar / energy
        psi0 = lam ** -1.5
        a0 = hbar / lam
        y0 = energy / lam * eps
        q, mass = float(self.physics['q']), float(self.physics['m'])
        self.cjq = q * hbar * psi0**2 * tau / (mass * lam * 2 * y0)
        self.cja = q**2 * a0 * psi0**2 * tau / (mass * 2 * y0)
        self.cpq = hbar * psi0**2 * tau / (mass * lam * 2 * y0)
        self.cpa = q * a0 * psi0**2 * tau / (mass * 2 * y0)
        self.b_background = float(self.physics['magneticFactor'])
        self.b0 = self.b_background
        self.psi0 = psi0
        self.energy0 = energy
        self.a0 = a0
        self.lambda0 = lam
        self.plot_x = self.x*lam
        self.plot_y = self.y*lam
        condition = str(self.problem['initialCondition'])
        # Campaigns can hold the initial orbit fixed while enlarging the box.
        center = self.problem.get('initialCenterLambda')
        radius = (float(np.linalg.norm(center))*lam if center is not None
                  else 0.375*float(self.plot_x[-1]))
        self.orbit_radius_a0 = radius if condition in ('cyclotron','paper_coherent') else None
        self.y0 = y0
        self.bscale = hbar/lam**2
        self.mu = 1/(float(self.physics['c'])**2*eps)
        self.v = self._row(source/'V'/'V.txt',self.n).reshape(self.ny,self.nx)
        self.initial_a = None
        self.diagnostics = {}
        result_path = source / 'result.json'
        if result_path.is_file():
            solver = json.loads(result_path.read_text()).get('solver', {})
            for key in ('gaussRelativeResidual', 'emRelativeResidual', 'qmRelativeResidual'):
                if key in solver:
                    self.diagnostics[key] = np.asarray(solver[key], dtype=float)
        if self.boundary == 'fixed_corrected':
            self.available = ('density', 'psi_real', 'psi_imag', 'phase', 'current',
                              'electric', 'electric_energy', 'magnetic_delta',
                              'magnetic_total', 'poynting', 'ham_quantum_gradient',
                              'ham_quantum_coupling', 'ham_quantum_potential',
                              'ham_electric', 'ham_magnetic', 'ham_total',
                              'ham_total_excess')
        else:
            self.available = ('density', 'psi_real', 'psi_imag', 'phase', 'electric')
            self.notes.append('Legacy spatial operators: current and magnetic perturbation omitted')

    def _row(self, path: Path, count: int) -> np.ndarray:
        row = np.loadtxt(path, delimiter=',').reshape(-1)
        if row.size != count or not np.isfinite(row).all():
            raise ValueError(f'Invalid SPHINX snapshot: {path}')
        return row

    def frame(self, index: int) -> Frame:
        step = self.steps[index]
        s = self._row(self.s_files[index], 2 * self.n)
        f = self._row(self.f_files[index], 6 * self.n)
        psi = (s[:self.n] + 1j * s[self.n:]).reshape(self.ny, self.nx)
        a = f[:3*self.n].reshape(3, self.ny, self.nx)
        y = f[3*self.n:].reshape(3, self.ny, self.nx)
        density = abs(psi)**2
        images = {'density': density, 'psi_real': psi.real, 'psi_imag': psi.imag,
                  'phase': np.angle(psi), 'electric': np.hypot(y[0], y[1]),
                  'electric_energy': 0.5*(y[0]**2+y[1]**2+y[2]**2)}
        vectors = {'electric': (-y[0], -y[1])}
        if self.boundary == 'fixed_corrected':
            q = float(self.physics['q'])
            hbar = float(self.physics['hbar'])
            eps = float(self.physics['eps_0'])
            dx = float(self.x[1] - self.x[0]); dy = float(self.y[1] - self.y[0])
            if self.initial_a is None:
                self.initial_a = a.copy() if step == 0 else None
                if self.initial_a is None:
                    raise ValueError('SPHINX corrected run requires step-zero snapshot')
            dax = a - self.initial_a
            delta_b = _forward(dax[1], dx, 1) - _forward(dax[0], dy, 0)
            drx = _forward(psi.real, dx, 1); dix = _forward(psi.imag, dx, 1)
            dry = _forward(psi.real, dy, 0); diy = _forward(psi.imag, dy, 0)
            jx = self.cpq*(psi.real*dix-psi.imag*drx)-self.cpa*density*a[0]
            jy = self.cpq*(psi.real*diy-psi.imag*dry)-self.cpa*density*a[1]
            images['current'] = density
            images['magnetic_delta'] = delta_b
            images['magnetic_total'] = self.b0+delta_b
            images['poynting'] = np.hypot(y[0], y[1])*abs(self.b0+delta_b)
            vectors['current'] = (jx, jy)
            vectors['poynting'] = (-y[1]*(self.b0+delta_b), y[0]*(self.b0+delta_b))
            lap_r = -_adjoint_forward(drx,dx,1)-_adjoint_forward(dry,dy,0)
            lap_i = -_adjoint_forward(dix,dx,1)-_adjoint_forward(diy,dy,0)
            hqg = -hbar**2*self.psi0**2/(4*float(self.physics['m'])*self.lambda0**2) * (psi.real*lap_r+psi.imag*lap_i)
            hqc = q*hbar*self.a0*self.psi0**2/(2*float(self.physics['m'])*self.lambda0) * (psi.imag*(a[0]*drx+a[1]*dry)-psi.real*(a[0]*dix+a[1]*diy))
            hqp = self.psi0**2/2 * (q**2*self.a0**2/(2*float(self.physics['m']))*np.sum(a*a,axis=0)*density + self.energy0/float(self.physics['m'])*self.v*density)
            he = 0.5*self.y0**2/eps*np.sum(y*y,axis=0)
            hm = 0.5*self.bscale**2/self.mu*(self.b0+delta_b)**2
            images.update(ham_quantum_gradient=hqg,ham_quantum_coupling=hqc,
                          ham_quantum_potential=hqp,ham_electric=he,
                          ham_magnetic=hm,ham_total=hqg+hqc+hqp+he+hm,
                          ham_total_excess=hqg+hqc+hqp+he+hm-
                              0.5*self.bscale**2/self.mu*self.b0**2)
        return Frame(images, vectors)


class OracleStaticRun(Run):
    def __init__(self, source: Path, allow_partial: bool = False):
        self.source = source.resolve()
        self.kind = 'oracle_static'
        self.p = loadmat(source / 'configuration.mat', simplify_cells=True)['p']
        self.name = str(self.p['name'])
        summary_path = source/'summary.json'
        self.summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        self.status = self.summary.get('status','partial')
        self.notes = []
        all_steps, all_files = _files(source, 'state', 'mat')
        checkpoint_path = source/'checkpoint.mat'
        if checkpoint_path.is_file():
            checkpoint = loadmat(checkpoint_path, simplify_cells=True)
            self.committed_step = int(checkpoint['step'])
            if int(self.summary.get('completedSteps',self.committed_step)) > self.committed_step:
                self.notes.append('TRANSFER MISMATCH: summary ahead of checkpoint; only checkpoint-backed frames used')
        else:
            self.committed_step = int(all_steps[-1])
            self.status = 'partial'
            self.notes.append('NO CHECKPOINT: saved states are rendered but not checkpoint-verified')
        if (self.status != 'completed' or self.committed_step < int(self.summary.get('requestedSteps',self.p['steps']))) and not allow_partial:
            raise ValueError('ORACLE static-B case is incomplete; pass --require-complete only for completed runs')
        keep = all_steps <= self.committed_step
        self.steps = all_steps[keep]
        self.files = [path for path, chosen in zip(all_files, keep) if chosen]
        expected = np.arange(0, self.committed_step+1, int(self.p['saveEvery']))
        if self.committed_step == int(self.p['steps']):
            expected = np.unique(np.r_[expected, int(self.p['steps'])])
        if not np.array_equal(self.steps, expected):
            if self.status == 'completed':
                raise ValueError('ORACLE static-B saved-state schedule is incomplete')
            self.notes.append('PARTIAL: saved-state schedule has gaps or an unscheduled endpoint')
        self.times = self.steps * float(self.p['dt'])
        width = float(self.p['halfWidthLB'])
        grid = int(self.p['grid'])
        self.x = np.linspace(-width, width, grid)[1:-1]/np.sqrt(float(self.p['B']))
        self.y = self.x.copy()
        self.plot_x = self.x
        self.plot_y = self.y
        self.orbit_radius_a0 = (np.sqrt(2*float(self.p['nbar'])/float(self.p['B']))
                                if str(self.p['kind']) == 'packet' else None)
        self.n = grid-2
        self.available = ('density', 'current', 'component_2', 'component_3',
                          'real_2', 'imag_2', 'real_3', 'imag_3', 'relative_phase',
                          'ham_dirac_rest','ham_dirac_kinetic','ham_dirac_total')
        diagnostics_path = source/'diagnostics.csv'
        if diagnostics_path.is_file():
            diag = np.genfromtxt(diagnostics_path, delimiter=',', names=True)
            self.diagnostics = {key: np.atleast_1d(diag[key]) for key in diag.dtype.names}
        else:
            self.diagnostics = {}
            self.notes.append('No stored diagnostics CSV; derived frame diagnostics remain available')
        manifest = source.parent / 'source_sha256.json'
        if manifest.is_file():
            for relative, expected_hash in json.loads(manifest.read_text()).items():
                source_file = source.parent/'source'/relative
                if not source_file.is_file():
                    self.notes.append('Frozen source file unavailable for verification: '+relative)
                    continue
                if sha256(source_file) != expected_hash:
                    raise ValueError(f'ORACLE frozen source mismatch: {relative}')
        else:
            self.notes.append('No frozen source manifest present')
        if self.status != 'completed' or self.committed_step < int(self.summary.get('requestedSteps',self.p['steps'])):
            self.status = 'partial'

    def frame(self, index: int) -> Frame:
        state = self.load_snapshot(index) if hasattr(self,'load_snapshot') else loadmat(self.files[index], simplify_cells=True)['snapshot']
        if int(state['step']) != int(self.steps[index]):
            raise ValueError('ORACLE snapshot step mismatch')
        u = np.asarray(state['u'])
        if u.shape != (self.n*self.n, 2) or not np.isfinite(u).all():
            raise ValueError('Invalid ORACLE static-B spinor')
        a = u[:,0].reshape((self.n,self.n), order='F').T
        b = u[:,1].reshape((self.n,self.n), order='F').T
        density = abs(a)**2+abs(b)**2
        current = np.conj(a)*b
        jx = (2*current.real).reshape(self.n,self.n)
        jy = (-2*current.imag).reshape(self.n,self.n)
        phase = np.angle(current)
        floor = max(float(np.max(abs(a)**2)),float(np.max(abs(b)**2)))*0.002
        phase = np.where((abs(a)**2 > floor)&(abs(b)**2 > floor), phase, np.nan)
        magnetic = float(self.p['B'])
        h = self.x[1]-self.x[0]
        order = int(self.p.get('spatialOrder',2))
        xphys = self.x
        xx,yy = np.meshgrid(xphys,xphys)
        pxa = -1j*_centered_zero(a,h,1,order)-magnetic*yy*a/2
        pya = -1j*_centered_zero(a,h,0,order)+magnetic*xx*a/2
        pxb = -1j*_centered_zero(b,h,1,order)-magnetic*yy*b/2
        pyb = -1j*_centered_zero(b,h,0,order)+magnetic*xx*b/2
        rest = abs(a)**2-abs(b)**2
        kinetic = np.real(np.conj(a)*(pxb+1j*pyb)+np.conj(b)*(pxa-1j*pya))
        return Frame(dict(density=density,current=density,
                          component_2=abs(a)**2,
                          component_3=abs(b)**2,
                          real_2=a.real,
                          imag_2=a.imag,
                          real_3=b.real,
                          imag_3=b.imag,
                          relative_phase=phase,ham_dirac_rest=rest,
                          ham_dirac_kinetic=kinetic,ham_dirac_total=rest+kinetic),
                     {'current': (jx,jy)})


class OracleCoupledRun(Run):
    def __init__(self, source: Path, allow_partial: bool = False):
        self.source = source.resolve()
        self.kind = 'oracle_coupled'
        self.sim = loadmat(source / 'configuration.mat', simplify_cells=True)['sim']
        self.name = source.name
        metadata_path = source/'metadata.mat'
        metadata = loadmat(metadata_path, simplify_cells=True)['metadata'] if metadata_path.is_file() else {}
        last_completed = int(metadata.get('lastCompletedStep', -1)) if isinstance(metadata, dict) else -1
        self.status = 'completed' if last_completed >= int(self.sim['steps']) else 'partial'
        self.steps, self.files = _files(source, 'state', 'mat')
        expected = np.arange(0, int(self.sim['steps'])+1, int(self.sim['output']['every']))
        if self.status == 'completed' and not np.array_equal(self.steps, expected):
            raise ValueError('Completed ORACLE coupled run has an incomplete saved-state schedule')
        self.times = float(self.sim['startTime']) + self.steps * float(self.sim['dt'])
        if self.status != 'completed' and not allow_partial:
            raise ValueError('ORACLE coupled run is not marked completed; pass --allow-partial')
        grid = np.asarray(self.sim['grid'], dtype=int)
        if grid.size != 3:
            raise ValueError('ORACLE coupled grid must have three dimensions')
        self.slice_z = int(grid[2]//2)
        origin = np.asarray(self.sim['origin'], dtype=float)
        spacing = np.asarray(self.sim['spacing'], dtype=float)
        self.x = origin[0] + np.arange(grid[0])*spacing[0]
        self.y = origin[1] + np.arange(grid[1])*spacing[1]
        self.plot_x = self.x
        self.plot_y = self.y
        self.orbit_radius_a0 = None
        self.physics = self.sim['physics']
        self.available = ('density','current','electric','magnetic_z','poynting',
                          'em_energy','component_2','component_3','ham_dirac_rest',
                          'ham_dirac_kinetic','ham_dirac_gauge','ham_electric',
                          'ham_magnetic','ham_total')
        self.notes = [f'Central z slice {self.slice_z}; full-domain density diagnostics are separate']
        self.diagnostics = {}

    def frame(self, index: int) -> Frame:
        state = self.load_snapshot(index) if hasattr(self,'load_snapshot') else loadmat(self.files[index], simplify_cells=True)['snapshot']
        if int(state['step']) != int(self.steps[index]):
            raise ValueError('ORACLE coupled snapshot step mismatch')
        psi_full = np.asarray(state['psi'])
        psi = psi_full[:,:,self.slice_z,:].transpose(1,0,2)
        Y = np.asarray(state['Y'],dtype=float)[:,:,self.slice_z,:].transpose(1,0,2)
        A = np.asarray(state['A'],dtype=float)
        if (psi.shape != (len(self.y),len(self.x),4) or
                Y.shape != (len(self.y),len(self.x),3) or
                A.shape != tuple(np.asarray(self.sim['grid'],dtype=int))+(3,) or
                not np.isfinite(psi).all() or not np.isfinite(Y).all() or
                not np.isfinite(A).all()):
            raise ValueError('ORACLE coupled spinor layout mismatch')
        density = np.sum(abs(psi)**2,axis=-1)
        alpha_x = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0],[1,0,0,0]])
        alpha_y = np.array([[0,0,0,-1j],[0,0,1j,0],[0,-1j,0,0],[1j,0,0,0]])
        speed = float(self.physics['c'])
        jx = speed*np.real(np.einsum('...a,ab,...b->...',psi.conj(),alpha_x,psi))
        jy = speed*np.real(np.einsum('...a,ab,...b->...',psi.conj(),alpha_y,psi))
        eps = float(self.physics['eps0'])
        ex,ey,ez = (-Y[:,:,component]/eps for component in range(3))
        spacing = np.asarray(self.sim['spacing'],dtype=float)
        d_ax = [_centered_projected(A[:,:,:,0],spacing[k],k) for k in range(3)]
        d_ay = [_centered_projected(A[:,:,:,1],spacing[k],k) for k in range(3)]
        d_az = [_centered_projected(A[:,:,:,2],spacing[k],k) for k in range(3)]
        bx = (d_az[1]-d_ay[2])[:,:,self.slice_z].T
        by = (d_ax[2]-d_az[0])[:,:,self.slice_z].T
        bz = (d_ay[0]-d_ax[1])[:,:,self.slice_z].T
        mu_inv = float(self.physics['c'])**2*eps
        sx = mu_inv*(ey*bz-ez*by)
        sy = mu_inv*(ez*bx-ex*bz)
        alpha_z = np.array([[0,0,1,0],[0,0,0,-1],[1,0,0,0],[0,-1,0,0]])
        alphas = (alpha_x,alpha_y,alpha_z)
        hkin = np.zeros_like(density)
        hgauge = np.zeros_like(density)
        for k,alpha in enumerate(alphas):
            derivative = _centered_projected(psi_full,spacing[k],k)[:,:,self.slice_z,:].transpose(1,0,2)
            hkin += np.real(np.einsum('...a,ab,...b->...',psi.conj(),alpha,
                                      -1j*float(self.physics['hbar'])*speed*derivative))
            hgauge += -float(self.physics['q'])*speed*A[:,:,self.slice_z,k].T * np.real(
                np.einsum('...a,ab,...b->...',psi.conj(),alpha,psi))
        mask = np.zeros_like(density)
        mask[1:-1,1:-1] = 1
        hkin *= mask
        hgauge *= mask
        hrest = float(self.physics['m'])*speed**2 * (
            abs(psi[:,:,0])**2+abs(psi[:,:,1])**2-abs(psi[:,:,2])**2-abs(psi[:,:,3])**2)
        he = 0.5*eps*(ex**2+ey**2+ez**2)
        hm = 0.5*mu_inv*(bx**2+by**2+bz**2)
        return Frame(dict(density=density,current=density,electric=np.hypot(ex,ey),
                          magnetic_z=bz,poynting=np.hypot(sx,sy),
                          em_energy=he+hm,
                          component_2=abs(psi[:,:,1])**2,component_3=abs(psi[:,:,2])**2,
                          ham_dirac_rest=hrest,ham_dirac_kinetic=hkin,
                          ham_dirac_gauge=hgauge,ham_electric=he,ham_magnetic=hm,
                          ham_total=hrest+hkin+hgauge+he+hm),
                     {'current':(jx,jy),'electric':(ex,ey),'poynting':(sx,sy)})


def open_run(source: str | Path, allow_partial: bool = False) -> Run:
    folder = Path(source).expanduser().resolve()
    if (folder / 'manifest.json').is_file() and (folder / 'S').is_dir():
        return SphinxRun(folder, allow_partial)
    if (folder / 'configuration.mat').is_file():
        config = loadmat(folder / 'configuration.mat', simplify_cells=True)
        if 'p' in config:
            return OracleStaticRun(folder, allow_partial)
        if 'sim' in config:
            if 'meta' in config and int(np.asarray(config['sim']['grid'])[2]) == 1:
                if config['meta'].get('solverVariant') in ('oracle_matlab_planar_v1','oracle_matlab_planar_v2','oracle_matlab_planar_v3','oracle_python_planar_v1'):
                    from .dirac_matlab import MatlabPlanarRun
                    return MatlabPlanarRun(folder,allow_partial)
                if config['meta'].get('solverVariant')=='sphinx_matched_forward_maxwell':
                    from .dirac_matched import MatchedDiracRun
                    return MatchedDiracRun(folder,allow_partial)
                from .dirac_longshot import DiracLongshotRun
                return DiracLongshotRun(folder, allow_partial)
            return OracleCoupledRun(folder, allow_partial)
    raise ValueError(f'Unsupported simulation folder: {folder}')
