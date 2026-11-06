"""Jacob Matthew Molina. Inception: 2026-11-05.

ORACLE four-branch atomic checkpoint and scalar-journal recovery. Adapted
from the tested SPHINX Python transaction design; no SPHINX import/runtime
is required. Source/config drift fails before recovery mutates outputs.
"""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from .evolution import State
from .maxwell import MurHistory

SCHEMA = 1
ARRAY_KEYS = ("psi", "A", "Y", "initial_longitudinal_y", "initial_harmonic_y", "previous_trace_y", "previous_valid")
TOTAL_KEYS = ("projection_work", "boundary_work", "wave_work", "reference_work")


def file_sha256(path, length=None):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        remaining = length
        while remaining is None or remaining:
            block = stream.read(1024*1024 if remaining is None else min(1024*1024, remaining))
            if not block:
                if remaining:
                    raise ValueError("Journal shorter than committed prefix")
                break
            digest.update(block)
            if remaining is not None:
                remaining -= len(block)
    return digest.hexdigest()


def _sync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_file(path, writer):
    """Flush complete bytes, replace atomically, then sync the containing folder."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Refusing to replace a symlink")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+b", dir=path.parent, prefix=f".{path.name}.",
                                         suffix=".partial", delete=False) as stream:
            temporary = Path(stream.name)
            writer(stream.file)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def atomic_json(path, payload):
    encoded = (json.dumps(payload, indent=2, allow_nan=False)+"\n").encode()
    atomic_file(path, lambda stream: stream.write(encoded))


@contextmanager
def run_lock(folder):
    """POSIX exclusive writer lock; inode remains after exit/crash."""
    import fcntl
    folder = Path(folder)
    if folder.is_symlink():
        raise ValueError("Run directory must not be a symlink")
    folder.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(folder/".production.lock", os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another writer owns this run") from error
        yield
    finally:
        os.close(descriptor)


def _scalars(values, name):
    if not isinstance(values, dict) or not values:
        raise ValueError(f"Missing {name}")
    for value in values.values():
        if not np.isscalar(value) or np.iscomplexobj(value) or not np.isfinite(value):
            raise ValueError(f"{name} must contain finite real scalars")


def _arrays(states):
    if len(states) != 4:
        raise ValueError("ORACLE campaign requires four synchronized branches")
    if any(state.mur is None for state in states):
        raise ValueError("All branches require initialized Mur history")
    arrays = {"psi": np.stack([s.psi for s in states]), "A": np.stack([s.a for s in states]),
              "Y": np.stack([s.y for s in states]),
              "initial_longitudinal_y": np.stack([s.mur.initial_longitudinal_y for s in states]),
              "initial_harmonic_y": np.stack([s.mur.initial_harmonic_y for s in states]),
              "previous_valid": np.array([s.mur.previous_trace_y is not None for s in states]),
              "previous_trace_y": np.stack([np.zeros_like(s.mur.initial_harmonic_y)
                  if s.mur.previous_trace_y is None else s.mur.previous_trace_y for s in states])}
    _validate_arrays(arrays)
    return arrays


def validate_states(states, expected_points=None):
    """Validate every control and Mur history before scientific publication."""
    arrays = _arrays(states)
    if expected_points is not None and arrays["psi"].shape[1] != expected_points:
        raise ValueError("Candidate grid differs from the configured grid")


def _validate_arrays(arrays):
    if set(arrays) != set(ARRAY_KEYS):
        raise ValueError("Checkpoint array schema differs")
    psi = arrays["psi"]
    if psi.ndim != 3 or psi.shape[0] != 4 or psi.shape[2] != 4 or psi.shape[1] < 1:
        raise ValueError("Checkpoint requires four [M,4] spinors")
    m = psi.shape[1]
    shapes = {"A": (4, m, 3), "Y": (4, m, 3), "initial_longitudinal_y": (4, 3*m),
              "initial_harmonic_y": (4, 3*m), "previous_trace_y": (4, 3*m), "previous_valid": (4,)}
    for key, array in arrays.items():
        if key != "psi" and array.shape != shapes[key]:
            raise ValueError(f"Checkpoint shape differs: {key}")
        if array.dtype.kind not in "fciub" or not np.all(np.isfinite(array)):
            raise ValueError(f"Nonfinite or nonnumerical checkpoint array: {key}")
        if key != "psi" and np.iscomplexobj(array):
            raise ValueError(f"Checkpoint field must be real: {key}")
    if arrays["previous_valid"].dtype.kind != "b":
        raise ValueError("Mur history availability must be boolean")


def _metadata(step, totals, info, initial, status):
    if isinstance(step, (bool, np.bool_)) or not isinstance(step, (int, np.integer)) or step < 0:
        raise ValueError("Checkpoint step must be a nonnegative integer")
    if set(totals) != set(TOTAL_KEYS):
        raise ValueError("Checkpoint cumulative-work schema differs")
    for value, name in [(totals, "work"), (info, "step diagnostics"), (initial, "initial diagnostics")]:
        _scalars(value, name)
    if status not in ("partial", "completed", "stopped"):
        raise ValueError("Invalid checkpoint status")
    return {"step": int(step), "totals": totals, "info": info, "initial": initial, "status": status}


def save_checkpoint(folder, states, step, totals, info, initial, status,
                    config_identity, source_identity, artifacts):
    """Commit all branches, history, cumulative work and flushed journal prefix.

    `artifacts` contains immutable configuration/source files and committed
    snapshots. Caller holds run_lock and publishes snapshots before checkpoint.
    """
    folder = Path(folder)
    arrays = _arrays(states)
    metadata = _metadata(step, totals, info, initial, status)
    if not config_identity or not source_identity:
        raise ValueError("Source and configuration identities are required")
    journal = folder/"coupled.csv"
    if journal.is_symlink():
        raise ValueError("Journal must not be a symlink")
    with journal.open("rb") as stream:
        os.fsync(stream.fileno())
        length = os.fstat(stream.fileno()).st_size
    metadata.update(schema=SCHEMA, config_identity=config_identity, source_identity=source_identity,
                    committed_bytes=length, journal_sha256=file_sha256(journal, length), artifacts=artifacts)
    encoded = json.dumps(metadata, sort_keys=True, allow_nan=False)
    atomic_file(folder/"checkpoint.npz", lambda stream: np.savez(stream, metadata=np.asarray(encoded), **arrays))


def load_checkpoint(folder, config_identity, source_identity, *, recover=True):
    """Validate every committed artifact before journal rollback/quarantine.

    A missing, shortened or changed journal/configuration/source/snapshot is
    rejected. Uncommitted snapshot files are preserved under recovery/ rather
    than mixed into a later analysis. No mismatch repairs itself silently.
    """
    folder = Path(folder)
    checkpoint, journal = folder/"checkpoint.npz", folder/"coupled.csv"
    if checkpoint.is_symlink() or journal.is_symlink():
        raise ValueError("Checkpoint/journal must not be symlinks")
    with np.load(checkpoint, allow_pickle=False) as record:
        if set(record.files) != set(ARRAY_KEYS) | {"metadata"}:
            raise ValueError("Checkpoint fields differ")
        metadata = json.loads(str(record["metadata"].item()))
        if metadata.get("schema") != SCHEMA:
            raise ValueError("Unsupported checkpoint schema")
        if metadata.get("config_identity") != config_identity or metadata.get("source_identity") != source_identity:
            raise ValueError("Source/configuration identity changed")
        arrays = {key: record[key].copy() for key in ARRAY_KEYS}
        _validate_arrays(arrays)
    _metadata(metadata["step"], metadata["totals"], metadata["info"], metadata["initial"], metadata["status"])
    if metadata["status"] == "stopped":
        raise ValueError("Screen-stopped run requires a reviewed new configuration")
    length = metadata["committed_bytes"]
    if isinstance(length, bool) or not isinstance(length, int) or length <= 0:
        raise ValueError("Invalid committed journal length")
    if file_sha256(journal, length) != metadata["journal_sha256"]:
        raise ValueError("Committed journal checksum differs")
    artifacts = metadata["artifacts"]
    required = {"configuration.json", "configuration.mat", "source_snapshot.zip", "state_000000000.mat"}
    if not isinstance(artifacts, dict) or not required <= artifacts.keys():
        raise ValueError("Required artifact receipt missing")
    for name, digest in artifacts.items():
        if name not in required and not re.fullmatch(r"state_\d{9}\.mat", name):
            raise ValueError("Unrecognized committed artifact path")
        path = folder/name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise ValueError(f"Committed artifact missing or changed: {name}")
    obsolete = []
    for path in folder.iterdir():
        match = re.fullmatch(r"state_(\d{9})\.mat", path.name)
        if match and path.name not in artifacts:
            if int(match[1]) <= metadata["step"] or path.is_symlink() or not path.is_file():
                raise ValueError("Unexpected snapshot inside committed prefix")
            obsolete.append(path)
    recovery = folder/"recovery"
    if recovery.is_symlink():
        raise ValueError("Recovery directory must not be a symlink")
    if recover:
        if obsolete:
            recovery.mkdir(exist_ok=True)
            for path in obsolete:
                with tempfile.NamedTemporaryFile(dir=recovery, prefix=path.stem+"_", suffix=".mat", delete=False) as stream:
                    target = Path(stream.name)
                os.replace(path, target)
            _sync_directory(recovery)
        with journal.open("r+b") as stream:
            stream.truncate(length)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(folder)
    states = []
    for branch in range(4):
        history = MurHistory(arrays["initial_longitudinal_y"][branch], arrays["initial_harmonic_y"][branch],
                             arrays["previous_trace_y"][branch] if arrays["previous_valid"][branch] else None)
        states.append(State(arrays["psi"][branch], arrays["A"][branch], arrays["Y"][branch], history))
    return tuple(states), metadata
