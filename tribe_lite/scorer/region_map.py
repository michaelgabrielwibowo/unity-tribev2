"""Brain region definitions - the canonical 'city map'."""

# Ordered mapping of region index to name. DO NOT REORDER - Unity C# code depends on stable indices.
REGION_MAP = {
    # FOREBRAIN — Cortex
    0: "Prefrontal Cortex",
    1: "Motor Cortex",
    2: "Somatosensory Cortex",
    3: "Auditory Cortex",
    4: "Visual Cortex (V1-V4)",
    5: "Insula",
    6: "Anterior Cingulate (ACC)",
    7: "Posterior Cingulate (PCC)",
    # FOREBRAIN — Basal Ganglia
    8: "Striatum",
    9: "Nucleus Accumbens",
    10: "Globus Pallidus",
    11: "Substantia Nigra",
    # FOREBRAIN — Limbic
    12: "Amygdala",
    13: "Hippocampus",
    14: "Septal Nuclei",
    # FOREBRAIN — Diencephalon
    15: "Thalamus",
    16: "Hypothalamus",
    17: "Pineal Gland",
    # MIDBRAIN
    18: "Superior Colliculus",
    19: "Inferior Colliculus",
    20: "VTA (Dopamine)",
    21: "Periaqueductal Gray",
    # HINDBRAIN
    22: "Pons",
    23: "Medulla",
    24: "Cerebellum (Hemispheres)",
    25: "Cerebellum (Vermis)",
}  # 26 regions total

# Reverse mapping: name → index
REGION_INDEX = {name: idx for idx, name in REGION_MAP.items()}

# Number of brain regions
NUM_REGIONS = len(REGION_MAP)


def get_region_name(idx: int) -> str:
    """Get region name by index."""
    return REGION_MAP.get(idx, f"Unknown Region {idx}")


def get_region_index(name: str) -> int:
    """Get region index by name."""
    return REGION_INDEX.get(name, -1)


def list_regions() -> list[tuple[int, str]]:
    """List all regions as (index, name) tuples."""
    return [(idx, REGION_MAP[idx]) for idx in range(NUM_REGIONS)]
