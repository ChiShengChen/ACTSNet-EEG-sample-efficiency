"""Channel montage definitions and name normalization for the ACTSNet pipeline.

Goal: map heterogeneous raw channel names (TUAB `EEG FP1-REF`, SEED-IV `Fp1`/`T7`)
onto a single canonical 10-20 vocabulary so the same montage selection works across
datasets, and the Paper 1 frontal-7 ablation uses identical channels everywhere.
"""

import re

# Canonical 19-channel 10-20 set (old TUAB-style temporal names T3/T4/T5/T6).
TEN_TWENTY_19 = [
    "FP1", "FP2", "F7", "F3", "FZ", "F4", "F8",
    "T3", "C3", "CZ", "C4", "T4",
    "T5", "P3", "PZ", "P4", "T6",
    "O1", "O2",
]

# Paper 1 deployable frontal montage (present in both TUAB and SEED-IV).
FRONTAL7 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8"]

MONTAGES = {
    "10_20_19": TEN_TWENTY_19,
    "frontal7": FRONTAL7,
}

# Synonyms → canonical. Modern (SEED-IV) temporal/parietal names map to the
# old 10-20 names so both datasets share one vocabulary.
_SYNONYMS = {
    "T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6",
}


def normalize_name(raw_name: str) -> str:
    """Normalize one raw channel label to canonical form.

    Strips a leading 'EEG ', a trailing '-REF'/'-LE' reference tag, whitespace,
    uppercases, then applies the modern→old synonym map.
    Returns '' for clearly non-EEG channels (EKG/PHOTIC/IBI/...) the caller drops.
    """
    n = raw_name.strip().upper()
    n = re.sub(r"^EEG\s+", "", n)
    n = re.sub(r"[-\s]*(REF|LE)$", "", n)
    n = n.strip()
    n = _SYNONYMS.get(n, n)
    return n


def build_index_map(raw_names, montage):
    """Return (indices, canonical_names) selecting `montage` channels from raw_names.

    Raises KeyError listing any montage channel absent from this recording, so the
    caller can skip recordings with an incomplete montage (logged, not silent).
    """
    target = MONTAGES[montage] if isinstance(montage, str) else list(montage)
    norm = [normalize_name(r) for r in raw_names]
    pos = {name: i for i, name in enumerate(norm) if name}  # first occurrence wins
    missing = [ch for ch in target if ch not in pos]
    if missing:
        raise KeyError(f"missing channels {missing}")
    idx = [pos[ch] for ch in target]
    return idx, list(target)
