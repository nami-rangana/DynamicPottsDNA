"""
extract_data.py
--------------------
Production script: distributes trajectory files across MPI ranks,
reads coordinates via netCDF4/scipy (bypasses broken MDTraj NetCDF reader),
computes all 6 backbone dihedrals for all kept nucleotides, and streams
results to per-trajectory Parquet files via PyArrow.

Run:
    mpirun -np 8 python extract_data.py
    # or on SLURM:
    srun --ntasks=8 python extract_data.py
"""

import os
import gc
import shutil
import tempfile
import numpy as np
import parmed as pmd
import mdtraj as md
import pyarrow as pa
import pyarrow.parquet as pq
from mpi4py import MPI

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
DATA_PTH   = "/orange/alberto.perezant/alberto.perezant/muABC/"
NC_DIR     = os.path.join(DATA_PTH, "nc")
TOP_DIR    = os.path.join(DATA_PTH, "top")
OUT_DIR    = os.path.join(DATA_PTH, "dihedrals_parquet")
CHUNK_SIZE = 500    # frames loaded into RAM per iteration
N_TERM     = 2      # terminal nucleotides to strip each end per strand

DNA_RESIDUE_NAMES = {
    "DA", "DT", "DG", "DC",
    "DA3", "DA5", "DT3", "DT5",
    "DG3", "DG5", "DC3", "DC5",
}

# Backbone dihedral definitions — atom names without padding
# (we .strip() both sides at lookup time)
DIHEDRAL_DEFS = [
    ("alpha", [(-1, "O3'"), ( 0, "P"  ), ( 0, "O5'"), ( 0, "C5'")]),
    ("beta",  [( 0, "P"  ), ( 0, "O5'"), ( 0, "C5'"), ( 0, "C4'")]),
    ("gamma", [( 0, "O5'"), ( 0, "C5'"), ( 0, "C4'"), ( 0, "C3'")]),
    ("delta", [( 0, "C5'"), ( 0, "C4'"), ( 0, "C3'"), ( 0, "O3'")]),
    ("eps",   [( 0, "C4'"), ( 0, "C3'"), ( 0, "O3'"), (+1, "P"  )]),
    ("zeta",  [( 0, "C3'"), ( 0, "O3'"), (+1, "P"  ), (+1, "O5'")]),
]
DIHEDRAL_NAMES = [d[0] for d in DIHEDRAL_DEFS]

# ─────────────────────────────────────────────────────────────
# FILE UTILITIES
# ─────────────────────────────────────────────────────────────

def get_file_names(path):
    out = []
    for root, _, files in os.walk(path):
        for f in files:
            out.append(os.path.join(root, f))
    return sorted(out)


def build_seq_from_tag(tag):
    return "GC" + tag[-2:] + tag + tag + tag + "GC"


def match_top_to_nc(tops, ncs):
    pairs = []
    for top in tops:
        tag = os.path.basename(top)[:4]
        for nc in ncs:
            base = os.path.basename(nc)
            if base.startswith(tag + ".") or base == tag + ".nc":
                pairs.append((nc, top, tag))
    return pairs

# ─────────────────────────────────────────────────────────────
# COORDINATE READER  (no MDTraj NetCDF — bypasses broken reader)
# ─────────────────────────────────────────────────────────────

def _read_nc_slice(nc_file, frame_slice):
    """
    Read a slice of frames from AMBER NetCDF.
    Returns np.ndarray (n_frames, n_atoms, 3) in NANOMETERS (Å ÷ 10).
    Tries netCDF4 first, falls back to scipy.
    """
    try:
        import netCDF4 as nc4
        with nc4.Dataset(nc_file, "r") as ds:
            raw = ds.variables["coordinates"][frame_slice]
            if hasattr(raw, "data"):
                raw = raw.data
            return np.asarray(raw, dtype=np.float32) / 10.0
    except ImportError:
        pass

    from scipy.io import netcdf_file
    with netcdf_file(nc_file, "r", mmap=False) as f:
        raw = f.variables["coordinates"].data[frame_slice]
        return np.asarray(raw, dtype=np.float32) / 10.0


def nc_n_frames(nc_file):
    """Total frame count without loading coordinates."""
    try:
        import netCDF4 as nc4
        with nc4.Dataset(nc_file, "r") as ds:
            return int(ds.variables["coordinates"].shape[0])
    except ImportError:
        from scipy.io import netcdf_file
        with netcdf_file(nc_file, "r", mmap=False) as f:
            return int(f.variables["coordinates"].data.shape[0])

# ─────────────────────────────────────────────────────────────
# TOPOLOGY BUILDER  (parmed → MDTraj, no OpenMM)
# ─────────────────────────────────────────────────────────────

def build_mdtraj_topology(pstruct):
    """
    Build mdtraj.Topology from parmed.Structure, DNA residues only.
    Returns (mdtraj_top, list_of_parmed_atom_indices).
    Atom names are stored as-is (may be padded); we strip at lookup.
    """
    top              = md.Topology()
    chain            = top.add_chain()
    prev_chain_id    = None
    dna_atom_indices = []   # into pstruct.atoms

    for res in pstruct.residues:
        if res.name not in DNA_RESIDUE_NAMES:
            continue

        chain_id = getattr(res, "chain", 0)
        if chain_id != prev_chain_id and prev_chain_id is not None:
            chain = top.add_chain()
        prev_chain_id = chain_id

        md_res = top.add_residue(res.name, chain, resSeq=res.number)

        for atom in res.atoms:
            sym = (atom.element_name or "C").strip() or "C"
            sym = {"EP": "VS", "LP": "VS", "": "C"}.get(sym, sym)
            try:
                element = md.element.Element.getBySymbol(sym)
            except KeyError:
                element = md.element.Element.getBySymbol("C")
            top.add_atom(atom.name, element, md_res)
            dna_atom_indices.append(atom.idx)

    return top, dna_atom_indices

# ─────────────────────────────────────────────────────────────
# DIHEDRAL INDEX TABLE  (built once per topology)
# ─────────────────────────────────────────────────────────────

def get_atom_index(top, res_idx, atom_name):
    """Atom index lookup with whitespace stripping on both sides."""
    target = atom_name.strip()
    for atom in top.residue(res_idx).atoms:
        if atom.name.strip() == target:
            return atom.index
    return None


def build_dihedral_table(top, strand_all, strand_keep):
    """
    Pre-compute atom-index quadruplets for every kept nucleotide.

    Returns:
        labels  : list of str  e.g. "S1_N3"
        quads   : np.ndarray (N_nt, 6, 4) — dtype int32; -1 = undefined
                  axis 1: [alpha, beta, gamma, delta, eps, zeta]
                  axis 2: [i0, i1, i2, i3]
    """
    labels = []
    quads  = []

    for nt_i, res_idx in enumerate(strand_keep):
        pos_in_full = strand_all.index(res_idx)
        for _, atom_specs in DIHEDRAL_DEFS:
            quad = []
            ok   = True
            for offset, aname in atom_specs:
                nbr_pos = pos_in_full + offset
                if not (0 <= nbr_pos < len(strand_all)):
                    ok = False; break
                nbr_res = strand_all[nbr_pos]
                aidx    = get_atom_index(top, nbr_res, aname)
                if aidx is None:
                    ok = False; break
                quad.append(aidx)
            quads.append(quad if ok else [-1, -1, -1, -1])

        labels.append(res_idx)   # store MDTraj res index for reference

    # shape: (n_nt * 6, 4)
    quads_arr = np.array(quads, dtype=np.int32)
    return labels, quads_arr   # quads_arr reshaped to (n_nt, 6, 4) later

# ─────────────────────────────────────────────────────────────
# VECTORISED DIHEDRAL COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_dihedrals_batch(xyz, quads_flat):
    """
    Compute dihedral angles for all valid quadruplets across all frames.

    xyz        : (n_frames, n_atoms, 3)  float32 nm
    quads_flat : (Q, 4)  int32  — rows with any -1 are skipped (→ NaN)

    Returns (n_frames, Q) float32 in degrees.
    """
    n_frames = xyz.shape[0]
    Q        = quads_flat.shape[0]
    out      = np.full((n_frames, Q), np.nan, dtype=np.float32)

    valid_mask = (quads_flat[:, 0] >= 0)   # True where no -1
    if not valid_mask.any():
        return out

    vq    = quads_flat[valid_mask]          # (V, 4)
    p0    = xyz[:, vq[:, 0], :]            # (n_frames, V, 3)
    p1    = xyz[:, vq[:, 1], :]
    p2    = xyz[:, vq[:, 2], :]
    p3    = xyz[:, vq[:, 3], :]

    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2

    n1 = np.cross(b1, b2)   # (n_frames, V, 3)
    n2 = np.cross(b2, b3)

    n1_norm = np.linalg.norm(n1, axis=-1, keepdims=True)  # (n_frames, V, 1)
    n2_norm = np.linalg.norm(n2, axis=-1, keepdims=True)

    good = (n1_norm[..., 0] > 1e-10) & (n2_norm[..., 0] > 1e-10)

    n1 = np.where(n1_norm > 1e-10, n1 / n1_norm, 0.0)
    n2 = np.where(n2_norm > 1e-10, n2 / n2_norm, 0.0)

    cos_a  = np.clip((n1 * n2).sum(-1), -1.0, 1.0)   # (n_frames, V)
    angles = np.degrees(np.arccos(cos_a)).astype(np.float32)

    sign = (np.cross(n1, n2) * b2).sum(-1)
    angles = np.where(sign < 0, -angles, angles)
    angles = np.where(good, angles, np.nan)

    out[:, valid_mask] = angles
    return out   # (n_frames, Q)

# ─────────────────────────────────────────────────────────────
# PARQUET OUTPUT
# ─────────────────────────────────────────────────────────────

def build_schema(col_names):
    fields = [pa.field("frame", pa.int32())]
    for c in col_names:
        fields.append(pa.field(c, pa.float32()))
    return pa.schema(fields)


def build_column_names(s1_keep, s2_keep, top):
    """
    Column names: alpha(S1_N3), beta(S1_N3), ..., zeta(S2_N16) etc.
    """
    cols = []
    for strand_label, kept in [("S1", s1_keep), ("S2", s2_keep)]:
        for i, res_idx in enumerate(kept):
            nt_tag = f"{strand_label}_N{i + N_TERM + 1}"
            for dname in DIHEDRAL_NAMES:
                cols.append(f"{dname}({nt_tag})")
    return cols


def write_batch(writer, frames_arr, angles_chunk, col_names):
    """
    Append one chunk to an open ParquetWriter.
    angles_chunk : (n_frames, n_nt * 6)  float32
    """
    batch_dict = {"frame": pa.array(frames_arr, type=pa.int32())}
    for ci, cname in enumerate(col_names):
        batch_dict[cname] = pa.array(
            angles_chunk[:, ci].tolist(), type=pa.float32()
        )
    writer.write_batch(pa.RecordBatch.from_pydict(batch_dict))

# ─────────────────────────────────────────────────────────────
# PER-TRAJECTORY WORKER
# ─────────────────────────────────────────────────────────────

def process_trajectory(nc_file, top_file, tag, rank):
    log = lambda msg: print(f"[Rank {rank:3d}] {msg}", flush=True)

    out_path = os.path.join(OUT_DIR, f"{tag}.parquet")
    if os.path.exists(out_path):
        log(f"SKIP {tag} — output already exists")
        return

    log(f"START {tag}  ({os.path.basename(nc_file)})")

    # ── Topology ──────────────────────────────────────────────────────
    pstruct = pmd.load_file(top_file)
    mdtraj_top, dna_atom_idx = build_mdtraj_topology(pstruct)

    n_res = mdtraj_top.n_residues
    seq   = build_seq_from_tag(tag)
    half  = len(seq)

    if n_res != 2 * half:
        log(f"WARNING: expected {2*half} DNA res, got {n_res}. "
            f"Overriding half={n_res//2}.")
        half = n_res // 2

    s1_all  = list(range(half))
    s2_all  = list(range(half, n_res))
    s1_keep = s1_all[N_TERM:-N_TERM]
    s2_keep = s2_all[N_TERM:-N_TERM]

    n_nt = len(s1_keep) + len(s2_keep)
    log(f"{tag}: {n_nt} nucleotides × 6 dihedrals per frame")

    # ── Dihedral index tables (built once) ────────────────────────────
    _, q_s1 = build_dihedral_table(mdtraj_top, s1_all, s1_keep)
    _, q_s2 = build_dihedral_table(mdtraj_top, s2_all, s2_keep)
    # shapes: (n_s1*6, 4) and (n_s2*6, 4)
    quads_flat = np.vstack([q_s1, q_s2])   # (n_nt*6, 4)

    col_names = build_column_names(s1_keep, s2_keep, mdtraj_top)
    assert len(col_names) == n_nt * 6

    schema = build_schema(col_names)
    dna_idx_arr = np.array(dna_atom_idx, dtype=np.int64)

    # ── Stream trajectory ─────────────────────────────────────────────
    n_frames_total = nc_n_frames(nc_file)
    log(f"{tag}: {n_frames_total} frames total, chunk={CHUNK_SIZE}")

    os.makedirs(OUT_DIR, exist_ok=True)
    writer       = pq.ParquetWriter(out_path, schema, compression="snappy")
    frame_offset = 0

    for chunk_start in range(0, n_frames_total, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_frames_total)

        # Read full-system coords, slice to DNA atoms
        coords_full = _read_nc_slice(nc_file, slice(chunk_start, chunk_end))
        # coords_full: (chunk, n_atoms_total, 3)
        coords_dna  = coords_full[:, dna_idx_arr, :]
        # coords_dna:  (chunk, n_dna_atoms, 3)

        n_chunk = coords_dna.shape[0]

        # Compute all dihedrals: (n_chunk, n_nt*6)
        angles = compute_dihedrals_batch(coords_dna, quads_flat)

        frames_arr = np.arange(chunk_start, chunk_start + n_chunk,
                               dtype=np.int32)
        write_batch(writer, frames_arr, angles, col_names)

        frame_offset += n_chunk
        log(f"{tag}: frames {chunk_start}–{chunk_start+n_chunk-1} done "
            f"({frame_offset}/{n_frames_total})")

        del coords_full, coords_dna, angles
        gc.collect()

    writer.close()
    log(f"DONE {tag} → {out_path}  ({frame_offset} frames, "
        f"{n_nt*6} columns)")

# ─────────────────────────────────────────────────────────────
# MPI DISPATCH
# ─────────────────────────────────────────────────────────────

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    tops  = get_file_names(TOP_DIR)
    ncs   = get_file_names(NC_DIR)
    pairs = match_top_to_nc(tops, ncs)

    if rank == 0:
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"Found {len(pairs)} trajectory/topology pairs.")
        print(f"Distributing over {size} MPI ranks.")
        for nc, top, tag in pairs:
            print(f"  {tag}: {os.path.basename(nc)}")
        print(flush=True)

    comm.Barrier()   # ensure OUT_DIR exists before workers start

    # Round-robin static distribution — each rank gets every Nth trajectory
    my_pairs = [p for i, p in enumerate(pairs) if i % size == rank]
    print(f"[Rank {rank:3d}] assigned {len(my_pairs)} trajectories: "
          f"{[p[2] for p in my_pairs]}", flush=True)

    for nc_file, top_file, tag in my_pairs:
        try:
            process_trajectory(nc_file, top_file, tag, rank)
        except Exception as e:
            import traceback
            print(f"[Rank {rank:3d}] ERROR on {tag}: {e}", flush=True)
            traceback.print_exc()
            # Continue to next trajectory rather than crashing the whole job

    comm.Barrier()
    if rank == 0:
        print(f"\nAll ranks finished. Output: {OUT_DIR}")

        # Print a summary of what was produced
        parquet_files = sorted(f for f in os.listdir(OUT_DIR)
                               if f.endswith(".parquet"))
        print(f"Parquet files written: {len(parquet_files)}")
        for f in parquet_files:
            path = os.path.join(OUT_DIR, f)
            size_mb = os.path.getsize(path) / 1e6
            pf = pq.read_metadata(path)
            print(f"  {f:<20s}  {pf.num_rows:>7d} frames  "
                  f"{pf.num_columns:>4d} cols  {size_mb:6.1f} MB")


if __name__ == "__main__":
    main()