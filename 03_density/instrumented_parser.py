"""Configurable parser brain + instrumented per-sentence parse.

Two things only:

1. ``ConfigurableEnglishParserBrain`` subclasses ``ParserBrain`` exactly
   like ``EnglishParserBrain``, but accepts a per-area ``{n, k}`` dict.
   No parser logic, word actions, fiber rules, or inhibition patterns
   change -- only per-area population size and cap size.

2. ``parse_sentence_instrumented`` is a copy of ``parser.parseHelper``
   (Algorithm 2 + Algorithm 3 FIBER_READOUT) with two additions:

   * The ``project_rounds`` loop snapshots winners after every
     ``parse_project`` call and records, per active area, the round at
     which two consecutive snapshots first match (the area's
     stabilization round).
   * The readout collects dependencies into a list instead of printing
     them, so the caller does not have to regex stdout.
"""

import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import parser as ac_parser  # noqa: E402
from parser import (  # noqa: E402
    AREAS, RECURRENT_AREAS,
    LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P,
    LEX_SIZE,
    LEXEME_DICT,
    ENGLISH_READOUT_RULES,
    ParserBrain,
)


# External-config name -> internal parser area constant.
AREA_NAME_MAP = {
    "LEX":   LEX,
    "DET":   DET,
    "SUBJ":  SUBJ,
    "OBJ":   OBJ,
    "VERB":  VERB,
    "ADJ":   ADJ,
    "ADV":   ADVERB,
    "PREP":  PREP,
    "PREPP": PREP_P,
}

# Reverse map for convergence reporting (parser-internal -> external).
INTERNAL_TO_EXTERNAL = {v: k for k, v in AREA_NAME_MAP.items()}


class ConfigurableEnglishParserBrain(ParserBrain):
    """Same as parser.EnglishParserBrain but with a per-area (n,k) config.

    The config dict uses the external area names from ``configs.py``
    (``LEX, DET, SUBJ, OBJ, VERB, ADJ, ADV, PREP, PREPP``).
    """

    def __init__(self, p, area_config,
                 default_beta=0.2,
                 LEX_beta=1.0,
                 recurrent_beta=0.05,
                 interarea_beta=0.5,
                 verbose=False):
        ParserBrain.__init__(
            self, p,
            lexeme_dict=LEXEME_DICT,
            all_areas=AREAS,
            recurrent_areas=RECURRENT_AREAS,
            initial_areas=[LEX, SUBJ, VERB],
            readout_rules=ENGLISH_READOUT_RULES,
        )
        self.verbose = verbose
        self.area_config = dict(area_config)

        # Translate external area names to parser internals.
        cfg = {AREA_NAME_MAP[name]: params
               for name, params in area_config.items()}

        # LEX: explicit area. Its inner connectome is n x n, so a large
        # LEX_n is expensive but otherwise harmless. We require
        # LEX_n >= LEX_SIZE * LEX_k so every word assembly fits.
        lex_n = cfg[LEX]["n"]
        lex_k = cfg[LEX]["k"]
        min_lex_n = LEX_SIZE * lex_k
        if lex_n < min_lex_n:
            raise ValueError(
                f"LEX n={lex_n} cannot hold LEX_SIZE={LEX_SIZE} word "
                f"assemblies of size k={lex_k} "
                f"(need n >= {min_lex_n})."
            )
        self.add_explicit_area(LEX, lex_n, lex_k, default_beta)

        # Other areas: order mirrors EnglishParserBrain for stable
        # connectome construction.
        for area_name in (SUBJ, OBJ, VERB, ADJ, PREP, PREP_P, DET, ADVERB):
            a = cfg[area_name]
            self.add_area(area_name, a["n"], a["k"], default_beta)

        # Plasticity: identical to EnglishParserBrain.
        custom_plasticities = defaultdict(list)
        for area in RECURRENT_AREAS:
            custom_plasticities[LEX].append((area, LEX_beta))
            custom_plasticities[area].append((LEX, LEX_beta))
            custom_plasticities[area].append((area, recurrent_beta))
            for other in RECURRENT_AREAS:
                if other == area:
                    continue
                custom_plasticities[area].append((other, interarea_beta))
        self.update_plasticities(area_update_map=custom_plasticities)

    # ---- Behavior duplicated from EnglishParserBrain ---------------

    def getProjectMap(self):
        proj_map = ParserBrain.getProjectMap(self)
        # "War of fibers" guard.
        if LEX in proj_map and len(proj_map[LEX]) > 2:
            raise Exception(
                "Got that LEX projecting into many areas: " + str(proj_map[LEX])
            )
        return proj_map

    def getWord(self, area_name, min_overlap=0.7):
        word = ParserBrain.getWord(self, area_name, min_overlap)
        if word:
            return word
        # DET special-case in EnglishParserBrain references DET_SIZE which
        # is not actually defined in parser.py; the code path was dead.
        # We return the parser's standard placeholder.
        return "<NON-WORD>"


# ----------------------------------------------------------------------
# Instrumented parse
# ----------------------------------------------------------------------

def _active_areas_from_proj_map(proj_map):
    active = set()
    for fa, tas in proj_map.items():
        active.add(fa)
        for ta in tas:
            active.add(ta)
    return active


def parse_sentence_instrumented(brain, sentence, project_rounds=20):
    """Run Algorithm 2 + FIBER_READOUT on ``sentence`` using ``brain``.

    Returns ``(dependencies, convergence)`` where:
      * ``dependencies`` is a list of ``[word_a, word_b, area_label]``
        triples produced by the readout.
      * ``convergence`` is a dict::

          {
            "num_words": int,
            "total_project_calls": int,
            "per_call": [
              {
                "word": str,
                "round_index": int,            # 0-based word position
                "active_areas": [str, ...],    # external names
                "per_area_rounds": {str: int},
                "rounds_to_stabilize_max": int,
              }, ...
            ],
          }

    Rounds-to-stabilize for an area = smallest ``r >= 2`` such that the
    area's k winners after round r equal the k winners after round r-1.
    If no such r is observed within ``project_rounds``, the recorded
    value is ``project_rounds`` (i.e. it never stabilized inside the
    allotted budget).
    """
    convergence = {
        "num_words": 0,
        "total_project_calls": 0,
        "per_call": [],
    }

    sentence_tokens = sentence.split(" ")

    for word_idx, word in enumerate(sentence_tokens):
        lexeme = LEXEME_DICT[word]
        brain.activateWord(LEX, word)

        for rule in lexeme["PRE_RULES"]:
            brain.applyRule(rule)

        # Same fix/unfix dance as parser.parseHelper.
        proj_map = brain.getProjectMap()
        for area in proj_map:
            if area not in proj_map[LEX]:
                brain.area_by_name[area].fix_assembly()
            elif area != LEX:
                brain.area_by_name[area].unfix_assembly()
                brain.area_by_name[area].winners = []

        # --- Instrumented projection loop ---------------------------
        # snapshots[area] = list of winners after each round (in order)
        snapshots = {}
        stabilized_at = {}
        active_total = set()

        for i in range(project_rounds):
            brain.parse_project()
            round_num = i + 1
            cur_proj_map = brain.getProjectMap()
            cur_active = _active_areas_from_proj_map(cur_proj_map)
            active_total.update(cur_active)

            for area in cur_active:
                cur = list(brain.area_by_name[area].winners)
                hist = snapshots.setdefault(area, [])
                hist.append(cur)
                if (area not in stabilized_at
                        and len(hist) >= 2
                        and hist[-1]
                        and hist[-1] == hist[-2]):
                    stabilized_at[area] = round_num

        per_area_rounds_internal = {
            area: stabilized_at.get(area, project_rounds)
            for area in active_total
        }
        per_area_rounds_ext = {
            INTERNAL_TO_EXTERNAL.get(a, a): r
            for a, r in per_area_rounds_internal.items()
        }
        rounds_max = (max(per_area_rounds_ext.values())
                      if per_area_rounds_ext else 0)

        convergence["per_call"].append({
            "word": word,
            "round_index": word_idx,
            "active_areas": sorted(per_area_rounds_ext.keys()),
            "per_area_rounds": per_area_rounds_ext,
            "rounds_to_stabilize_max": rounds_max,
        })
        convergence["total_project_calls"] += 1

        for rule in lexeme["POST_RULES"]:
            brain.applyRule(rule)

    convergence["num_words"] = len(sentence_tokens)

    # --- Readout (FIBER_READOUT, mirrors parser.parseHelper) --------
    brain.disable_plasticity = True
    for area in AREAS:
        brain.area_by_name[area].unfix_assembly()

    dependencies = []

    def read_out(area, mapping):
        to_areas = mapping[area]
        brain.project({}, {area: list(to_areas)})
        this_word = brain.getWord(LEX)
        for to_area in to_areas:
            if to_area == LEX:
                continue
            brain.project({}, {to_area: [LEX]})
            other_word = brain.getWord(LEX)
            dependencies.append([this_word, other_word, to_area])
        for to_area in to_areas:
            if to_area != LEX:
                read_out(to_area, mapping)

    activated_fibers = brain.getActivatedFibers()
    read_out(VERB, activated_fibers)

    return dependencies, convergence
