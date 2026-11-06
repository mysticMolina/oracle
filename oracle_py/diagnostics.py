"""Jacob Matthew Molina. Inception: 2026-11-05.

Source single-species planar compatible diagnostics. Moment, chi and free-
negative-energy quantities retain their MATLAB definitions and limitations;
they are not reinterpreted as interacting particle or pair-production counts.
"""

import numpy as np

from .core import dirac_matrices
from .matter import charge_density, dirac_current
from .operators import spatial_hamiltonian


def diagnostics(state, ctx, origin, charge=-1., mass=1., hbar=1.,
                background_charge=None, background_a=None, background_b=None,
                units=None):
    """Return all compatible planar fields from diracMaxwellDiagnostics.

    Centroid values remain source API fields; this function makes no plot.
    `physical-compton` enables the source's m=hbar=|q|=1 model screens.
    """
    psi, a, y = np.asarray(state.psi), np.asarray(state.a), np.asarray(state.y)
    m = len(ctx.matter.free)
    if psi.shape != (m, 4) or a.shape != (m, 3) or y.shape != (m, 3):
        raise ValueError("expected one packed [M,4] spinor and [M,3] fields")
    if not all(np.all(np.isfinite(v)) for v in (psi, a, y)) or not np.isrealobj(a) or not np.isrealobj(y):
        raise ValueError("state must be finite, with real fields")
    origin = np.asarray(origin, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError("origin must be a finite three-vector")
    if not np.all(np.isfinite([charge, mass, hbar])) or min(mass, hbar) <= 0:
        raise ValueError("invalid physical constants")
    background_a = np.zeros_like(a) if background_a is None else np.asarray(background_a)
    background_b = np.zeros_like(a) if background_b is None else np.asarray(background_b)
    if any(v.shape != (m, 3) or not np.isrealobj(v) or not np.all(np.isfinite(v))
           for v in (background_a, background_b)):
        raise ValueError("background fields must be finite real [M,3]")
    volume, c, eps = ctx.volume, ctx.c, ctx.eps0
    u = psi.ravel(order="F")
    h = spatial_hamiltonian(a, ctx.matter, charge, c, hbar)
    density = np.sum(np.abs(psi)**2, axis=1)
    number = np.sum(density)
    norm = volume * np.vdot(u, u).real
    spatial_energy = volume * np.vdot(u, h @ u).real
    rest_energy = volume * mass * c*c * np.sum(np.abs(psi)**2 * np.array([1, 1, -1, -1]))
    delta_b = ctx.field.curl @ (a-background_a).ravel(order="F")
    total_b = delta_b + background_b.ravel(order="F")
    electric_energy = volume * np.sum(y*y) / (2 * eps)
    perturbation_energy = volume * eps * c*c * np.dot(delta_b, delta_b) / 2
    magnetic_energy = volume * eps * c*c * np.dot(total_b, total_b) / 2
    rho = charge_density(psi, charge, background_charge)
    g = ctx.field.free_gradient.T @ y.ravel(order="F") - rho[ctx.matter.free]
    reference = np.linalg.norm(rho[ctx.matter.free])
    absolute = np.linalg.norm(g)
    d = {"norm": norm, "charge": charge * norm, "spatialEnergy": spatial_energy,
         "restEnergy": rest_energy, "electricEnergy": electric_energy,
         "perturbationMagneticEnergy": perturbation_energy, "magneticEnergy": magnetic_energy,
         "emEnergy": electric_energy + perturbation_energy,
         "totalEnergy": spatial_energy + rest_energy + electric_energy + perturbation_energy,
         "backgroundInclusiveEnergy": spatial_energy + rest_energy + electric_energy + magnetic_energy,
         "gaussLinf": np.max(np.abs(g)), "gaussAbsolute": absolute,
         "gaussReferenceNorm": reference, "gaussRelativeDefined": float(reference > 0),
         "gaussRelative": absolute / reference if reference > 0 else 0.,
         "divBLinf": np.max(np.abs(ctx.field.magnetic_divergence @ total_b)),
         "maximumDensity": np.max(density)}
    nx, ny, _ = ctx.grid
    wave = np.fft.fft2(psi.reshape((nx, ny, 4), order="F"), axes=(0, 1))
    power = np.sum(np.abs(wave)**2, axis=2)
    frequencies = [np.fft.ifftshift(np.arange(-(size//2), (size+1)//2)) / size
                   for size in (nx, ny)]
    tail = (np.abs(frequencies[0][:, None]) > .25) | (np.abs(frequencies[1][None, :]) > .25)
    d["highKFraction"] = np.sum(power[tail]) / max(np.sum(power), np.finfo(float).tiny)
    d["kineticPerNormDefined"] = float(norm > 0)
    d["kineticPerNorm"] = (rest_energy+spatial_energy) / norm - mass*c*c if norm > 0 else 0.
    if norm > 0 and charge != 0:
        x, yy = np.meshgrid(origin[0] + np.arange(nx)*ctx.spacing[0],
                           origin[1] + np.arange(ny)*ctx.spacing[1], indexing="ij")
        current = dirac_current(psi, charge, c)
        weight = max(number, np.finfo(float).tiny)
        d["xMean"] = np.dot(density, x.ravel(order="F")) / weight
        d["yMean"] = np.dot(density, yy.ravel(order="F")) / weight
        d["vxMean"] = np.sum(current[:, 0]) / (charge*weight)
        d["vyMean"] = np.sum(current[:, 1]) / (charge*weight)
        momentum2 = np.zeros(m)
        for axis in (0, 1):
            mechanical = -1j*hbar*(ctx.matter.derivatives[axis] @ psi) - charge*a[:, axis, None]*psi
            momentum2 += np.sum(np.abs(mechanical)**2, axis=1)
        d["momentumRms"] = np.sqrt(np.sum(momentum2) / weight)
        electric = -y / eps
        b = total_b.reshape((m, 3), order="F")
        half = np.max(np.abs(origin[:2]))
        collar = (np.abs(x.ravel(order="F")) > .8*half) | (np.abs(yy.ravel(order="F")) > .8*half)
        d["boundaryMassFraction"] = np.sum(density[collar]) / weight
        d["Emax"] = np.max(np.linalg.norm(electric, axis=1))
        d["totalBmax"] = np.max(np.linalg.norm(b, axis=1))
        if units == "physical-compton":
            if mass != 1 or hbar != 1 or abs(charge) != 1:
                raise ValueError("source physical-compton screens require m=hbar=|q|=1")
            localu = np.sqrt(momentum2 / np.maximum(density, 1e-300)) / c
            gamma = np.sqrt(1+localu*localu)
            chi = (gamma*np.linalg.norm(electric, axis=1) + localu*c*np.abs(b[:, 2])) / c**3
            active = density > 1e-6*np.max(density)
            order = np.argsort(chi, kind="stable")
            cumulative = np.cumsum(density[order]) / weight
            index = min(np.searchsorted(cumulative, .999), m-1)
            d["chiProxy999"] = chi[order[index]]
            d["chiProxyActiveMax"] = np.max(chi[active])
            invariant = np.sqrt(np.maximum(np.sum(electric**2, axis=1)-c*c*b[:, 2]**2, 0)) / c**3
            d["electricInvariantOverSchwinger"] = np.max(invariant[active])
            kx = 2*np.pi*frequencies[0]/ctx.spacing[0]
            ky = 2*np.pi*frequencies[1]/ctx.spacing[1]
            px = np.sin(kx*ctx.spacing[0])[:, None]/ctx.spacing[0]
            py = np.sin(ky*ctx.spacing[1])[None, :]/ctx.spacing[1]
            energy = np.sqrt(c**4+c*c*(px*px+py*py))
            spectral = wave.reshape((m, 4), order="F")
            alpha, beta = dirac_matrices()
            pxgrid = np.broadcast_to(px, (nx, ny)).ravel(order="F")
            pygrid = np.broadcast_to(py, (nx, ny)).ravel(order="F")
            hw = c*c*(spectral @ beta.T) + c*(pxgrid[:, None]*(spectral @ alpha[0].T)
                                                           + pygrid[:, None]*(spectral @ alpha[1].T))
            negative = (spectral-hw/energy.ravel(order="F")[:, None]) / 2
            total = max(np.sum(power), np.finfo(float).tiny)
            d["freeNegativeFraction"] = np.sum(np.abs(negative)**2)/total
            d["freeGroupVelocity"] = np.sum(power*c*c*px/energy*np.cos(kx*ctx.spacing[0])[:, None])/total
    return {key: float(value) for key, value in d.items()}


def boundary_probes(state, ctx, background_a=None, background_b=None, background_charge=None, charge=-1.):
    """Source inset TE/TM characteristics and continuum flux (not port work)."""
    from .maxwell import initialize_mur, mur_reference_field
    nx, ny, _ = ctx.grid
    m = nx*ny
    background_a = np.zeros((m, 3)) if background_a is None else background_a
    background_b = np.zeros((m, 3)) if background_b is None else background_b
    rho = charge_density(state.psi, charge, background_charge)
    history = initialize_mur(rho, ctx, state.mur)
    longitudinal = mur_reference_field(rho, ctx, history)
    electric = (-state.y/ctx.eps0).reshape((nx, ny, 3), order="F")
    wave = (-(state.y.ravel(order="F")-longitudinal)/ctx.eps0).reshape((nx, ny, 3), order="F")
    delta = (ctx.field.curl @ (state.a-background_a).ravel(order="F")).reshape((nx, ny, 3), order="F")
    total = delta + np.asarray(background_b).reshape((nx, ny, 3), order="F")
    result = {}
    for face, name in enumerate(("Left", "Right", "Bottom", "Top")):
        sign = -1 if face in (0, 2) else 1
        if face < 2:
            i = 1 if face == 0 else nx-2
            midpoint = max(1, ny//2)-1
            dbz = (delta[i-1, :, 2]+delta[i, :, 2])/2
            dbt = sign*(delta[i-1, :, 1]+delta[i, :, 1])/2
            et, ez = sign*wave[i, :, 1], wave[i, :, 2]
            btotal = (total[i-1, :, 2]+total[i, :, 2])/2
            bttotal = sign*(total[i-1, :, 1]+total[i, :, 1])/2
            electric_t, electric_z, h = sign*electric[i, :, 1], electric[i, :, 2], ctx.spacing[1]
        else:
            j = 1 if face == 2 else ny-2
            midpoint = max(1, nx//2)-1
            dbz = (delta[:, j-1, 2]+delta[:, j, 2])/2
            dbt = -sign*(delta[:, j-1, 0]+delta[:, j, 0])/2
            et, ez = -sign*wave[:, j, 0], wave[:, j, 2]
            btotal = (total[:, j-1, 2]+total[:, j, 2])/2
            bttotal = -sign*(total[:, j-1, 0]+total[:, j, 0])/2
            electric_t, electric_z, h = -sign*electric[:, j, 0], electric[:, j, 2], ctx.spacing[0]
        prefix = "probe"+name
        result[prefix+"OutgoingTE"] = et[midpoint]+ctx.c*dbz[midpoint]
        result[prefix+"IncomingTE"] = et[midpoint]-ctx.c*dbz[midpoint]
        result[prefix+"OutgoingTM"] = ez[midpoint]-ctx.c*dbt[midpoint]
        result[prefix+"IncomingTM"] = ez[midpoint]+ctx.c*dbt[midpoint]
        weights = np.ones_like(electric_z)
        weights[[0, -1]] = .5
        te = np.sum(electric_t[:-1]*btotal[:-1])*h
        tm = np.sum(weights*electric_z*bttotal)*h
        result["flux"+name] = ctx.eps0*ctx.c**2*(te-tm)
    return {name: float(value) for name, value in result.items()}
