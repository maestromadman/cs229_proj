


CONFIGS = {
    "baseline": {
        
        "LEX":   {"n": 10000, "k": 100},
        "VERB":  {"n": 10000, "k": 100},
        "SUBJ":  {"n": 10000, "k": 100},
        "OBJ":   {"n": 10000, "k": 100},
        "DET":   {"n": 10000, "k": 100},
        "ADJ":   {"n": 10000, "k": 100},
        "ADV":   {"n": 10000, "k": 100},
        "PREP":  {"n": 10000, "k": 100},
        "PREPP": {"n": 10000, "k": 100},
    },

    "experiment_A_verb_enlarged": {
        # Only VERB enlarged -- isolates the hub effect.
        "LEX":   {"n": 10000, "k": 100},
        "VERB":  {"n": 20000, "k": 200},
        "SUBJ":  {"n": 10000, "k": 100},
        "OBJ":   {"n": 10000, "k": 100},
        "DET":   {"n": 10000, "k": 100},
        "ADJ":   {"n": 10000, "k": 100},
        "ADV":   {"n": 10000, "k": 100},
        "PREP":  {"n": 10000, "k": 100},
        "PREPP": {"n": 10000, "k": 100},
    },

    "experiment_B_closed_class_shrunk": {
        # Only DET and PREP shrunk -- isolates the closed-class effect.
        "LEX":   {"n": 10000, "k": 100},
        "VERB":  {"n": 10000, "k": 100},
        "SUBJ":  {"n": 10000, "k": 100},
        "OBJ":   {"n": 10000, "k": 100},
        "DET":   {"n": 4000,  "k": 40},
        "ADJ":   {"n": 10000, "k": 100},
        "ADV":   {"n": 10000, "k": 100},
        "PREP":  {"n": 4000,  "k": 40},
        "PREPP": {"n": 10000, "k": 100},
    },

    "experiment_C_proportional": {
        # Full graded scaling -- main hypothesis.
        "LEX":   {"n": 10000, "k": 100},
        "VERB":  {"n": 15000, "k": 150},
        "SUBJ":  {"n": 12000, "k": 120},
        "OBJ":   {"n": 12000, "k": 120},
        "DET":   {"n": 6000,  "k": 60},
        "ADJ":   {"n": 8000,  "k": 80},
        "ADV":   {"n": 8000,  "k": 80},
        "PREP":  {"n": 6000,  "k": 60},
        "PREPP": {"n": 10000, "k": 100},
    },

    "experiment_D_inverted": {
        # Inverted -- negative control, should perform worst.
        "LEX":   {"n": 10000, "k": 100},
        "VERB":  {"n": 4000,  "k": 40},
        "SUBJ":  {"n": 10000, "k": 100},
        "OBJ":   {"n": 10000, "k": 100},
        "DET":   {"n": 20000, "k": 200},
        "ADJ":   {"n": 10000, "k": 100},
        "ADV":   {"n": 10000, "k": 100},
        "PREP":  {"n": 20000, "k": 200},
        "PREPP": {"n": 10000, "k": 100},
    },
}


SHORT_NAMES = {
    "baseline":                       "baseline",
    "experiment_A_verb_enlarged":     "exp_A_verb_enlarged",
    "experiment_B_closed_class_shrunk": "exp_B_closed_shrunk",
    "experiment_C_proportional":      "exp_C_proportional",
    "experiment_D_inverted":          "exp_D_inverted",
}


REQUIRED_AREAS = ("LEX", "VERB", "SUBJ", "OBJ", "DET", "ADJ",
                  "ADV", "PREP", "PREPP")


def validate_config(name, config):
    """Assert k/n == 0.01 for every area; assert all areas are present."""
    missing = [a for a in REQUIRED_AREAS if a not in config]
    if missing:
        raise ValueError(
            f"Config {name!r}: missing required areas {missing}"
        )
    for area, params in config.items():
        if "n" not in params or "k" not in params:
            raise ValueError(
                f"Config {name!r}, area {area!r}: needs both n and k"
            )
        n = params["n"]
        k = params["k"]
        if n <= 0 or k <= 0:
            raise ValueError(
                f"Config {name!r}, area {area!r}: n={n}, k={k} must be > 0"
            )
        # Exact-ratio assertion. Use integer arithmetic so we are not
        # bitten by float roundoff: k/n == 0.01  <=>  100 * k == n.
        if 100 * k != n:
            raise ValueError(
                f"Config {name!r}, area {area!r}: k/n = {k}/{n} = "
                f"{k / n:.6f}, expected 0.01 (i.e. n == 100*k)"
            )


def validate_all():
    for name, cfg in CONFIGS.items():
        validate_config(name, cfg)


if __name__ == "__main__":
    validate_all()
    print(f"All {len(CONFIGS)} configs valid (k/n == 0.01 for every area).")
    for name in CONFIGS:
        print(f"  - {name}  (short: {SHORT_NAMES[name]})")
