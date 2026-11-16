import random, sys

# Deterministic-agreement corpus built from the SAME lexicon as the minimal pairs in
# tools/grammar.cpp. Every sentence is grammatical, so a model that has learned number agreement
# can score above chance and one that only memorises word pairs cannot. The pairs themselves are
# never emitted as training sentences: the corpus is generated from the lexicon, the pairs are
# hand-written, so the test is generalisation to novel combinations of known words.

NOUN = {
    "key": "keys", "box": "boxes", "child": "children", "man": "men", "team": "teams",
    "file": "files", "car": "cars", "house": "houses", "water": "waters", "coach": "coaches",
    "room": "rooms", "table": "tables", "park": "parks", "woman": "women",
}
SG = list(NOUN)
PL = [NOUN[k] for k in NOUN]
DET_SG = ["the", "a", "this", "that"]
DET_PL = ["the", "these", "those"]
V3SG = {"turn": "turns", "work": "works", "leave": "leaves", "open": "opens", "find": "finds"}
VBASE = {v: k for k, v in V3SG.items()}
# past forms, needed so that the relative-clause verbs in the minimal pairs are in vocabulary
VPAST = {"found": "find", "opened": "open", "saw": "see", "left": "leave"}
BE3SG, BEBASE = "is", "are"
HAVE3SG, HAVEBASE = "has", "have"
WAS3SG, WASBASE = "was", "were"
PREP = ["in", "on", "with", "near"]
REL = ["that", "who"]
REL_VERB = list(V3SG.values()) + list(VPAST)
SUBJ_VERB = ["the", "a"]

def subject(rng, pl):
    return rng.choice(DET_PL if pl else DET_SG), rng.choice(PL if pl else SG)

def vp(rng, pl, aux=False):
    r = rng.random()
    if r < 0.35:
        return [rng.choice(list(VBASE.values())) if pl else rng.choice(list(V3SG.values()))]
    if r < 0.55:
        return [HAVEBASE if pl else HAVE3SG]
    if r < 0.75:
        return [BEBASE if pl else BE3SG]
    return [WASBASE if pl else WAS3SG]

def sentence(rng):
    pl = rng.random() < 0.5
    det, n = subject(rng, pl)
    out = [det, n]
    if rng.random() < 0.35:
        out += [rng.choice(PREP), "the", rng.choice(SG if rng.random() < 0.5 else PL)]
    if rng.random() < 0.4:
        # relative clause: the embedded subject has its own number and is a distractor, which is
        # what makes agreement-at-a-distance hard
        epl = rng.random() < 0.5
        out += [rng.choice(REL), "the", rng.choice(PL if epl else SG), rng.choice(REL_VERB)]
    out += vp(rng, pl)
    return " ".join(out) + " ."

def one_of(rng):
    pl = rng.random() < 0.5
    return "one of the %s %s ." % (rng.choice(PL if pl else SG), rng.choice(list(V3SG.values())))

def main():
    seed, n, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = random.Random(seed)
    with open(out, "w") as f:
        for i in range(n):
            f.write((one_of(rng) if i % 9 == 0 else sentence(rng)) + "\n")

main()
