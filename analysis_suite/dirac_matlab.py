"""Jacob Matthew Molina, 2026-10-16. Read MATLAB planar campaign outputs.

MATLAB public arrays are [Ny,Nx,Nz,component]; the established Dirac analysis
interface uses [Nx,Ny,Nz,component]. No evolution is performed by this reader.
"""
import numpy as np
from scipy.io import loadmat
from .dirac_matched import MatchedDiracRun
from .adapters import _forward


class MatlabPlanarRun(MatchedDiracRun):
    def __init__(self, source, allow_partial=False):
        cfg = loadmat(source / 'configuration.mat', simplify_cells=True)
        units = cfg['meta'].get('units', 'atomic')
        if units not in ('atomic', 'physical-compton'):
            raise ValueError('Unknown MATLAB planar unit system')
        self.atomic_conversion = 1.0
        if cfg['sim'].get('operatorProfile') != 'compatible-forward':
            raise ValueError('Unsupported MATLAB planar operator profile')
        super().__init__(source, allow_partial)
        self.times = float(self.sim['startTime']) + self.steps * float(self.sim['dt'])
        implementation = 'Python CPU parity solver' if self.meta.get('solverVariant') == 'oracle_python_planar_v1' else 'MATLAB production solver'
        self.notes += [implementation + '; public y/x arrays converted explicitly to analysis x/y order.']
        self.notes += [f"Boundary: {self.meta.get('boundaryCondition')}; initial electrostatic closure: {self.meta.get('longitudinal', 'grounded')}."]
        if 'boundaryValidation' in self.meta:
            self.notes += [str(self.meta['boundaryValidation'])]
        if (source/'FAILED_VALIDATION.txt').exists():
            self.notes += ['FAILED scientific adoption: see source FAILED_VALIDATION.txt; retained for diagnosis.']
        if units == 'physical-compton':
            self._convert_compton_to_atomic()

    def _convert_compton_to_atomic(self):
        # Fixed reference units, independent of c_factor. epsilon is transformed
        # with the same pinned scale, retaining the source's exact alpha.
        c0 = float(self.sim.get('unitReference', {}).get('cAtomic', 137.03599917759013))
        self.atomic_conversion = c0
        self.x = self.x / c0; self.y = self.y / c0
        self.plot_x = self.x; self.plot_y = self.y
        self.times = self.times / c0**2
        self.sim['spacing'] = np.asarray(self.sim['spacing'], float) / np.array([c0, c0, 1.])
        self.sim['origin'] = np.asarray(self.sim['origin'], float) / np.array([c0, c0, 1.])
        self.sim['dt'] = float(self.sim['dt']) / c0**2
        self.sim['startTime'] = float(self.sim['startTime']) / c0**2
        self.physics['c'] = float(self.physics['c']) * c0
        self.physics['eps0'] = float(self.physics['eps0']) / c0
        self.meta['B'] = float(self.meta['B']) * c0**2
        self.sim['backgroundCharge'] = np.asarray(self.sim['backgroundCharge']) * c0**3
        for key in ('backgroundA', 'backgroundB'):
            if key in self.sim:
                self.sim[key] = np.asarray(self.sim[key]) * (c0 if key == 'backgroundA' else c0**2)
        for key, value in self.saved.items():
            if key.startswith('probe'):
                scale = c0**3
            elif key.startswith('flux'):
                scale = c0**5
            elif 'energy' in key.lower() or key.endswith('Work') or key in ('spatialEnergy', 'restEnergy'):
                scale = c0**3  # planar energy per invariant z length
            elif key in ('kineticPerNorm', 'prescribedKinetic', 'freeKinetic'):
                scale = c0**2
            else:
                scale = dict(time=c0**-2, norm=c0, charge=c0, xMean=1/c0,
                             yMean=1/c0, vxMean=c0, vyMean=c0, momentumRms=c0,
                             Emax=c0**3, totalBmax=c0**2, gaussLinf=c0**3,
                             gaussAbsolute=c0**3, gaussReferenceNorm=c0**3,
                             projectionNorm=c0**2, boundaryWork=c0**3,
                             cumulativeBoundaryWork=c0**3).get(key, 1.)
            self.saved[key] = value * scale
        self.meta['sourceUnits'] = 'physical-compton'; self.meta['units'] = 'atomic'
        self.notes += ['Compton inputs converted to atomic units for the established plots: '
                       'x/c0, t/c0^2, psi*c0^(3/2), A*c0, Y*c0^2; '
                       'planar integrated energy scales as c0^3. c0 is fixed, not model c.']

    def electromagnetic_fields(self, index):
        state = self.load_snapshot(index)
        nx, ny, _ = map(int, self.sim['grid'])
        a = state['A'][:, :, 0, :].transpose(1, 0, 2)
        background_a = np.asarray(self.sim['backgroundA']).reshape(ny, nx, 1, 3)[:, :, 0, :]
        background_b = np.asarray(self.sim['backgroundB']).reshape(ny, nx, 1, 3)[:, :, 0, :]
        da = a - background_a
        dx, dy = self.x[1]-self.x[0], self.y[1]-self.y[0]
        db = np.stack([_forward(da[..., 2], dy, 0), -_forward(da[..., 2], dx, 1),
                       _forward(da[..., 1], dx, 1)-_forward(da[..., 0], dy, 0)], axis=-1)
        electric = -state['Y'][:, :, 0, :].transpose(1, 0, 2) / self.physics['eps0']
        return electric, db, db + background_b

    def frame(self, index):
        frame = super().frame(index)
        electric, db, magnetic = self.electromagnetic_fields(index)
        eps = float(self.physics['eps0']); c = float(self.physics['c'])
        he = eps*np.sum(electric**2, axis=-1)/2
        hb = eps*c*c*np.sum(db**2, axis=-1)/2
        flux = eps*c*c*np.cross(electric, magnetic)
        frame.images.update(electric=np.linalg.norm(electric, axis=-1), ham_electric=he,
                            ham_magnetic=hb, em_energy=he+hb, poynting=np.linalg.norm(flux[..., :2], axis=-1))
        frame.images['ham_total'] = sum(frame.images[k] for k in
                                       ('ham_dirac_rest', 'ham_dirac_kinetic', 'ham_dirac_gauge',
                                        'ham_electric', 'ham_magnetic'))
        frame.vectors['poynting'] = (flux[..., 0], flux[..., 1])
        return frame

    def load_snapshot(self, index, branch=0):
        data = loadmat(self.files[index], simplify_cells=True)
        if branch == 0:
            state = data['snapshot']
        else:
            state = np.asarray(data['allStates'], dtype=object).ravel()[branch]
        step = int(data['snapshot']['step'])
        if step != int(self.steps[index]):
            raise ValueError('MATLAB snapshot step mismatch')
        expected = float(self.sim['startTime']) + step * float(self.sim['dt'])
        expected *= self.atomic_conversion**2
        if not np.isclose(float(data['snapshot']['time']), expected, rtol=1e-12, atol=1e-14):
            raise ValueError('MATLAB snapshot time mismatch')
        nx, ny, nz = map(int, self.sim['grid'])
        out = {'step': step}
        for name, count in (('psi', 4), ('A', 3), ('Y', 3)):
            public = np.asarray(state[name]).reshape((ny, nx, nz, count))
            out[name] = public.transpose(1, 0, 2, 3)
            c0 = self.atomic_conversion
            out[name] = out[name] * dict(psi=c0**1.5, A=c0, Y=c0**2)[name]
            if not np.isfinite(out[name]).all():
                raise ValueError('Nonfinite MATLAB snapshot')
        return out
