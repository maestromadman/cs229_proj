
from collections import namedtuple

TEMPLATE_DESCRIPTIONS = {
    1:  "N V-INTRANS",
    2:  "N V N",
    3:  "D N V-INTRANS",
    4:  "D N V N",
    5:  "D N V D N",
    6:  "ADJ N V N",
    7:  "D ADJ N D ADJ N",
    8:  "PRO V PRO",
    9:  "N V-INTRANS ADVERB",
    10: "N ADVERB V-INTRANS",
    11: "N V-INTRANS ADVERB",
    12: "N ADVERB V N",
    13: "N V-INTRANS PP",
    14: "N V-INTRANS PP PP",
    15: "N V N PP",
    16: "N PP V N",
    17: "N COPULA N",
    18: "N COPULA ADJ",
    19: "complex copula",
    20: "chained adjectives",
}


def deps_for_intrans(subj, verb, subj_det=None, subj_adj=None,
                     adverb=None, pp1=None, pp2=None,
                     subj_pp=None):
   
    d = {(verb, subj, "SUBJ")}
    if subj_det:
        d.add((subj, subj_det, "DET"))
    if subj_adj:
        d.add((subj, subj_adj, "ADJ"))
    if adverb:
        d.add((verb, adverb, "ADVERB"))
    if pp1:
        p, n = pp1
        d.add((verb, n, "PREP_P"))
        d.add((n, p, "PREP"))
    if pp2:
        
        p, n = pp2
        first_n = pp1[1]
        d.add((first_n, n, "PREP_P"))
        d.add((n, p, "PREP"))
    if subj_pp:
        p, n = subj_pp
        d.add((subj, n, "PREP_P"))
        d.add((n, p, "PREP"))
    return d


def deps_for_trans(subj, verb, obj,
                   subj_det=None, subj_adj=None,
                   obj_det=None, obj_adj=None,
                   adverb=None,
                   obj_pp=None):
    d = {(verb, subj, "SUBJ"), (verb, obj, "OBJ")}
    if subj_det:
        d.add((subj, subj_det, "DET"))
    if subj_adj:
        d.add((subj, subj_adj, "ADJ"))
    if obj_det:
        d.add((obj, obj_det, "DET"))
    if obj_adj:
        d.add((obj, obj_adj, "ADJ"))
    if adverb:
        d.add((verb, adverb, "ADVERB"))
    if obj_pp:
        p, n = obj_pp
        d.add((obj, n, "PREP_P"))
        d.add((n, p, "PREP"))
    return d


def deps_for_copula_noun(subj, cop, pred_noun,
                         subj_det=None, subj_adj=None,
                         pred_det=None):
    
    d = {(cop, subj, "SUBJ"), (cop, pred_noun, "OBJ")}
    if subj_det:
        d.add((subj, subj_det, "DET"))
    if subj_adj:
        d.add((subj, subj_adj, "ADJ"))
    if pred_det:
        d.add((pred_noun, pred_det, "DET"))
    return d


def deps_for_copula_adj(subj, cop, pred_adj,
                        subj_det=None, subj_adjs=()):
    
    d = {(cop, subj, "SUBJ"), (cop, pred_adj, "ADJ")}
    if subj_det:
        d.add((subj, subj_det, "DET"))
    for a in subj_adjs:
        d.add((subj, a, "ADJ"))
    return d




SENTENCES = []  


def add(tid, sent, deps):
    SENTENCES.append({
        "template_id": tid,
        "template_desc": TEMPLATE_DESCRIPTIONS[tid],
        "sentence": sent,
        "expected_deps": [list(t) for t in sorted(deps)],
    })



for s, v in [("dogs", "run"), ("cats", "fly"), ("mice", "run"), ("people", "fly"),
             ("dogs", "fly"), ("cats", "run"), ("mice", "fly"), ("people", "run"),
             ("man", "run"), ("woman", "fly")]:
    add(1, f"{s} {v}", deps_for_intrans(s, v))


T2 = [("dogs", "chase", "cats"), ("cats", "chase", "mice"),
      ("mice", "love", "dogs"), ("people", "bite", "cats"),
      ("dogs", "love", "mice"), ("cats", "bite", "people"),
      ("mice", "saw", "dogs"), ("people", "saw", "mice"),
      ("dogs", "saw", "people"), ("cats", "love", "dogs")]
for s, v, o in T2:
    add(2, f"{s} {v} {o}", deps_for_trans(s, v, o))


T3 = [("the", "dogs", "run"), ("the", "cats", "fly"),
      ("a", "man", "run"), ("a", "woman", "fly"),
      ("the", "mice", "run"), ("the", "people", "fly"),
      ("a", "man", "fly"), ("a", "woman", "run"),
      ("the", "man", "run"), ("the", "woman", "fly")]
for d, s, v in T3:
    add(3, f"{d} {s} {v}", deps_for_intrans(s, v, subj_det=d))


T4 = [
    ("the", "dogs", "chase", None, "cats"),
    ("a", "man", "saw", None, "woman"),
    (None, "dogs", "chase", "the", "cats"),
    (None, "people", "love", "a", "man"),
    ("the", "cats", "love", None, "mice"),
    ("a", "woman", "bite", None, "man"),
    (None, "mice", "love", "the", "dogs"),
    (None, "cats", "bite", "a", "woman"),
    ("the", "people", "saw", None, "man"),
    (None, "dogs", "saw", "a", "man"),
]
for sd, s, v, od, o in T4:
    toks = [t for t in [sd, s, v, od, o] if t is not None]
    add(4, " ".join(toks),
        deps_for_trans(s, v, o, subj_det=sd, obj_det=od))


T5 = [("the", "man", "saw", "the", "woman"),
      ("the", "dogs", "chase", "the", "cats"),
      ("a", "man", "love", "a", "woman"),
      ("the", "cats", "love", "the", "mice"),
      ("a", "woman", "saw", "a", "man"),
      ("the", "people", "chase", "the", "dogs"),
      ("a", "dogs", "bite", "a", "cats"),
      ("the", "mice", "love", "the", "dogs"),
      ("a", "cats", "saw", "a", "mice"),
      ("the", "woman", "bite", "the", "man")]
for sd, s, v, od, o in T5:
    add(5, f"{sd} {s} {v} {od} {o}",
        deps_for_trans(s, v, o, subj_det=sd, obj_det=od))


T6 = [
    ("big", "dogs", "chase", None, "cats"),
    ("bad", "cats", "love", None, "mice"),
    ("big", "man", "saw", None, "woman"),
    ("bad", "woman", "bite", None, "man"),
    (None, "dogs", "chase", "big", "cats"),
    (None, "cats", "love", "bad", "mice"),
    (None, "man", "saw", "big", "woman"),
    (None, "mice", "bite", "bad", "dogs"),
    ("big", "people", "saw", None, "cats"),
    (None, "people", "love", "bad", "dogs"),
]
for sa, s, v, oa, o in T6:
    toks = [t for t in [sa, s, v, oa, o] if t is not None]
    add(6, " ".join(toks),
        deps_for_trans(s, v, o, subj_adj=sa, obj_adj=oa))


T7 = [
    ("the", "big", "man", "saw", "the", "bad", "woman"),
    ("a", "bad", "man", "bite", "a", "big", "woman"),
    ("the", "big", "dogs", "chase", "the", "bad", "cats"),
    ("a", "bad", "cats", "love", "a", "big", "mice"),
    ("the", "big", "people", "saw", "the", "bad", "dogs"),
    ("a", "big", "man", "love", "a", "bad", "woman"),
    ("the", "bad", "woman", "bite", "the", "big", "man"),
    ("a", "big", "mice", "chase", "a", "bad", "cats"),
    ("the", "bad", "dogs", "love", "the", "big", "cats"),
    ("a", "big", "people", "saw", "a", "bad", "mice"),
]
for sd, sa, s, v, od, oa, o in T7:
    add(7, f"{sd} {sa} {s} {v} {od} {oa} {o}",
        deps_for_trans(s, v, o,
                       subj_det=sd, subj_adj=sa,
                       obj_det=od, obj_adj=oa))


T8 = [("man", "saw", "woman"), ("woman", "saw", "man"),
      ("dogs", "love", "mice"), ("cats", "bite", "people"),
      ("people", "saw", "dogs"), ("mice", "chase", "cats"),
      ("man", "bite", "woman"), ("woman", "love", "man"),
      ("dogs", "chase", "people"), ("cats", "love", "mice")]
for s, v, o in T8:
    add(8, f"{s} {v} {o}", deps_for_trans(s, v, o))


T9 = [
    (None, "dogs", "run"), (None, "cats", "fly"),
    ("the", "mice", "run"), ("a", "man", "fly"),
    (None, "people", "fly"), ("the", "woman", "run"),
    (None, "mice", "run"), ("the", "dogs", "fly"),
    ("a", "woman", "fly"), (None, "people", "run"),
]
for d, s, v in T9:
    toks = [t for t in [d, s, v, "quickly"] if t is not None]
    add(9, " ".join(toks),
        deps_for_intrans(s, v, subj_det=d, adverb="quickly"))


T10 = [
    (None, "dogs", "run"), (None, "cats", "fly"),
    ("the", "mice", "run"), ("a", "man", "fly"),
    (None, "people", "fly"), ("the", "woman", "run"),
    (None, "mice", "run"), ("the", "dogs", "fly"),
    ("a", "woman", "fly"), (None, "people", "run"),
]
for d, s, v in T10:
    toks = [t for t in [d, s, "quickly", v] if t is not None]
    add(10, " ".join(toks),
        deps_for_intrans(s, v, subj_det=d, adverb="quickly"))


T11 = [
    (None, "big", "dogs", "run"), (None, "bad", "cats", "fly"),
    ("the", "big", "mice", "run"), ("a", "bad", "man", "fly"),
    (None, "big", "people", "fly"), ("the", "bad", "woman", "run"),
    (None, "bad", "mice", "run"), ("the", "big", "dogs", "fly"),
    ("a", "big", "woman", "fly"), (None, "bad", "people", "run"),
]
for d, a, s, v in T11:
    toks = [t for t in [d, a, s, v, "quickly"] if t is not None]
    add(11, " ".join(toks),
        deps_for_intrans(s, v, subj_det=d, subj_adj=a, adverb="quickly"))


T12 = [
    (None, "dogs", "chase", None, "cats"),
    (None, "cats", "love", None, "mice"),
    ("the", "man", "saw", "the", "woman"),
    ("a", "woman", "bite", "a", "man"),
    (None, "mice", "bite", None, "dogs"),
    (None, "people", "love", None, "cats"),
    ("the", "dogs", "saw", "the", "cats"),
    ("a", "man", "love", "a", "woman"),
    ("the", "cats", "chase", None, "mice"),
    (None, "people", "love", None, "dogs"),
]
for sd, s, v, od, o in T12:
    toks = [t for t in [sd, s, "quickly", v, od, o] if t is not None]
    add(12, " ".join(toks),
        deps_for_trans(s, v, o, subj_det=sd, obj_det=od, adverb="quickly"))


T13 = [
    (None, "dogs", "run", "in", "cats"),
    (None, "cats", "fly", "in", "mice"),
    ("the", "mice", "run", "of", "dogs"),
    ("a", "man", "fly", "in", "woman"),
    (None, "people", "run", "in", "cats"),
    ("the", "woman", "fly", "of", "man"),
    (None, "mice", "run", "of", "dogs"),
    ("the", "dogs", "fly", "in", "cats"),
    ("a", "woman", "fly", "of", "man"),
    (None, "people", "run", "of", "mice"),
]
for d, s, v, p, n in T13:
    toks = [t for t in [d, s, v, p, n] if t is not None]
    add(13, " ".join(toks),
        deps_for_intrans(s, v, subj_det=d, pp1=(p, n)))


T15 = [
    (None, "dogs", "chase", None, "cats", "of", "mice"),
    (None, "cats", "love", None, "mice", "of", "dogs"),
    ("the", "man", "saw", "the", "woman", "of", "dogs"),
    ("a", "woman", "bite", "a", "man", "of", "mice"),
    ("the", "people", "saw", None, "cats", "of", "dogs"),
    (None, "mice", "love", "the", "dogs", "of", "cats"),
    (None, "cats", "bite", None, "mice", "of", "people"),
    ("a", "man", "love", "a", "woman", "of", "mice"),
    (None, "dogs", "chase", "the", "cats", "of", "dogs"),
    (None, "people", "love", None, "cats", "of", "dogs"),
]
for sd, s, v, od, o, p, n in T15:
    toks = [t for t in [sd, s, v, od, o, p, n] if t is not None]
    add(15, " ".join(toks),
        deps_for_trans(s, v, o,
                       subj_det=sd, obj_det=od,
                       obj_pp=(p, n)))


T16 = [
    (None, "dogs", "in", "cats", "chase", "mice"),
    ("the", "man", "of", "woman", "saw", "dogs"),
    (None, "cats", "in", "mice", "love", "dogs"),
    ("a", "man", "of", "woman", "saw", "cats"),
    (None, "mice", "in", "dogs", "love", "cats"),
    (None, "people", "of", "cats", "saw", "dogs"),
    ("the", "dogs", "in", "cats", "chase", "mice"),
    ("a", "woman", "of", "man", "bite", "dogs"),
    (None, "cats", "of", "people", "love", "mice"),
    (None, "dogs", "in", "mice", "saw", "cats"),
]
for sd, s, p, n, v, o in T16:
    toks = [t for t in [sd, s, p, n, v, o] if t is not None]
    add(16, " ".join(toks),
        deps_for_trans(s, v, o,
                       subj_det=sd,
                       
                       ))
    
    SENTENCES[-1]["expected_deps"] = [list(t) for t in sorted(
        set(tuple(x) for x in SENTENCES[-1]["expected_deps"])
        | {(s, n, "PREP_P"), (n, p, "PREP")}
    )]


T17 = [
    (None, "dogs", "are", None, "cats"),
    ("the", "dogs", "are", None, "mice"),
    ("a", "man", "are", None, "woman"),
    (None, "cats", "are", None, "dogs"),
    ("the", "mice", "are", None, "people"),
    ("a", "woman", "are", None, "man"),
    (None, "people", "are", None, "dogs"),
    ("the", "man", "are", None, "woman"),
    (None, "cats", "are", "the", "mice"),
    (None, "dogs", "are", "a", "cats"),
]
for sd, s, cop, pd, p in T17:
    toks = [t for t in [sd, s, cop, pd, p] if t is not None]
    add(17, " ".join(toks),
        deps_for_copula_noun(s, cop, p, subj_det=sd, pred_det=pd))


T18 = [
    (None, "dogs", "are", "big"),
    (None, "cats", "are", "bad"),
    ("the", "dogs", "are", "big"),
    ("a", "man", "are", "bad"),
    (None, "people", "are", "big"),
    ("the", "mice", "are", "bad"),
    (None, "cats", "are", "big"),
    ("the", "woman", "are", "bad"),
    ("a", "man", "are", "big"),
    (None, "mice", "are", "big"),
]
for sd, s, cop, a in T18:
    toks = [t for t in [sd, s, cop, a] if t is not None]
    add(18, " ".join(toks),
        deps_for_copula_adj(s, cop, a, subj_det=sd))


T19 = [
    (None, "big", "dogs", "are", "bad"),
    (None, "bad", "cats", "are", "big"),
    ("the", "big", "mice", "are", "bad"),
    ("a", "bad", "man", "are", "big"),
    (None, "big", "people", "are", "bad"),
    ("the", "bad", "woman", "are", "big"),
    (None, "bad", "mice", "are", "big"),
    ("the", "big", "dogs", "are", "bad"),
    ("a", "big", "woman", "are", "bad"),
    (None, "bad", "people", "are", "big"),
]
for sd, sa, s, cop, pa in T19:
    toks = [t for t in [sd, sa, s, cop, pa] if t is not None]
    add(19, " ".join(toks),
        deps_for_copula_adj(s, cop, pa, subj_det=sd, subj_adjs=(sa,)))




ACTIVE_TEMPLATES = [tid for tid in range(1, 21) if tid not in (14, 20)]
EXCLUDED_TEMPLATES = (14, 20)

assert len(SENTENCES) == 180, f"Expected 180 sentences, got {len(SENTENCES)}"

from collections import Counter
counts = Counter(s["template_id"] for s in SENTENCES)
for tid in ACTIVE_TEMPLATES:
    assert counts[tid] == 10, f"Template {tid} has {counts[tid]} sentences"
for tid in EXCLUDED_TEMPLATES:
    assert tid not in counts, f"Excluded template {tid} should have no sentences"


if __name__ == "__main__":
    for s in SENTENCES[:5]:
        print(s)
    print("...")
    print(f"Total: {len(SENTENCES)}")
