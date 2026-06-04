
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import parser as ac_parser  
from parser import ( 
    AREAS, RECURRENT_AREAS,
    LEX, DET, SUBJ, OBJ, VERB, ADJ, ADVERB, PREP, PREP_P,
    LEX_SIZE,
    LEXEME_DICT,
    ENGLISH_READOUT_RULES,
    ParserBrain,
)



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


INTERNAL_TO_EXTERNAL = {v: k for k, v in AREA_NAME_MAP.items()}


class ConfigurableEnglishParserBrain(ParserBrain):
    

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

        
        cfg = {AREA_NAME_MAP[name]: params
               for name, params in area_config.items()}

        
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

        
        for area_name in (SUBJ, OBJ, VERB, ADJ, PREP, PREP_P, DET, ADVERB):
            a = cfg[area_name]
            self.add_area(area_name, a["n"], a["k"], default_beta)

        
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
        
        return "<NON-WORD>"



def _active_areas_from_proj_map(proj_map):
    active = set()
    for fa, tas in proj_map.items():
        active.add(fa)
        for ta in tas:
            active.add(ta)
    return active


def parse_sentence_instrumented(brain, sentence, project_rounds=20):
   
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

        
        proj_map = brain.getProjectMap()
        for area in proj_map:
            if area not in proj_map[LEX]:
                brain.area_by_name[area].fix_assembly()
            elif area != LEX:
                brain.area_by_name[area].unfix_assembly()
                brain.area_by_name[area].winners = []

        
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
