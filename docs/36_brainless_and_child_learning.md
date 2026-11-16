# 36 · Brains without brains, and how children learn (persona panel)

Eleven research agents ran as personas (developmental psychologist, slime-mould biophysicist,
plant electrophysiologist, comparative neurobiologist, immunologist, consciousness scientist, and
others) against arXiv, PubMed/DOI and publisher pages. Everything below is sourced; contested
claims are flagged.

## Part 1 · How a child learns (and what our metric misses)

| fact | number |
|---|---|
| adult words per 12-h day | ~12,300 (Gilkerson et al. 2017, 3,213 recordings) |
| extrapolated words per year | ~4.5M (my arithmetic from the above) |
| words by age 5 | ~20M |
| adult English lemmas known | 42,000 (Brysbaert et al. 2016) |
| infant word segmentation from transitional probability | after **2 minutes** of passive listening (Saffran, Aslin & Newport 1996) |
| data gap vs LLMs | **4-5 orders of magnitude** (Frank 2023) |
| CHILDES vs LM corpus | 7.5M vs 852M tokens (Chang & Bergen 2022) |
| returns on input | children **accelerate**, LMs are constant-proportional (Frank 2026) |
| phoneme-level boundary representations | usable at ~180k words (Goriely et al. 2025) |

**The sharpest result**: Chang & Bergen (2022) show LMs rely far more on word frequency than
children do, and Frank (2026) shows children's learning *accelerates* with exposure while LM loss
improves at a constant proportional rate. Our single number (corpus top-1) cannot see any of this.

What to change (ranked):

1. **Per-word age-of-acquisition curves.** Fit a sigmoid to each word's surprisal, define AoA at a
   fixed surprisal cutoff, and test whether AoA accelerates with exposure. This is the measurement
   that separates a child-like learner from a bag of bigrams.
2. **Boundary readout.** Train the character stack to minimise next-unit error, then read word
   boundaries at error peaks and report boundary F1. Infants do this in two minutes; we have never
   measured it.
3. **Discourse-local chunking** (Chang et al. 2024): local ordering helps, global developmental
   curricula do not.
4. **Multi-cue boundary fusion**: transitional probability + utterance edges + stress, not TP alone.
5. **Novelty-gated plasticity**: rare informative items get disproportionate consolidation
   (consistent with one-trial fast mapping, Medina et al. 2011).

## Part 2 · Computation without neurons

| organism | mechanism | measured numbers |
|---|---|---|
| *Physarum polycephalum* | flux-based tube thickening `dD/dt = f(|Q|) - D/tau` | shortest path through a maze (Nakagaki 2000); Tokyo rail network parity in cost, efficiency, fault tolerance (Tero 2010) |
| *Physarum* | externalised memory (avoid own slime) | Y-maze 39/40 correct; masking the trail drops U-trap success 96% -> 33% (Reid et al. 2012) |
| *Dionaea muscipula* | action-potential counting | 2 APs -> close, 3 -> jasmonate, 5 -> digestion (Bohm et al. 2016) |
| *Arabidopsis* | glutamate-triggered Ca2+ wave | ~1 mm/s to distal leaves in minutes (Toyota et al. 2018) |
| *Mimosa pudica* | habituation | reported 28-day retention (Gagliano 2014), **not replicated** (2020) |
| *Trichoplax* | no neurons; neuropeptide oscillations | 0.002-0.02 Hz locomotor rhythm, ~4% neurosecretory-like cells (2017-2024) |
| sponges | ATP/glutamate whole-body contraction | coordinated without nerves (2022-2025) |
| *Cassiopea* | sleep-like state | reduced pulsation, rebound after deprivation (Nath et al. 2017) |
| box jellyfish | operant conditioning in the rhopalia | obstacle avoidance improves with collision pairing (2023) |
| immune system | clonal selection, shape space | 10^-3 mutations/bp/generation, memory cells lasting decades |
| mycorrhizal networks | carbon transfer measured, "sharing" not | critique: Karst et al. 2023 |

Contested or unsupported: Hydra associative learning (no controlled study found), plant "learning",
Mimosa replication, Physarum "anticipation" (single study, no replication), directed mycorrhizal
transfer, ctenophore "first nervous system" homology.

## Part 3 · Consciousness as engineering constraints

| theory | testable claim | 2025 evidence |
|---|---|---|
| IIT | posterior "hot zone" sustains content | posterior decoding supported (Cogitate) |
| GWT | prefrontal broadcast enables report | prefrontal sustained activity largely **absent** |
| Higher-order | a second representation of the first | PFC disruption reduces metacognition |
| Attention schema | awareness = internal model of attention | predicts specific mislocalisation |

Measurable proxies for a machine: meta-d'/d' > 0 (Fleming & Lau 2014), confidence calibration
ECE < 0.05, post-error slowing, and accurate state report. None of these need a theory of
consciousness; all of them need a *self-model* and a *broadcast*.

## Part 4 · What CortexMesh should copy (with cost)

| mechanism | source | cost | where |
|---|---|---|---|
| importance-scaled Hebb | SI/MAS + immune memory | 1 float/synapse | done (D42, +19 %) |
| threshold-count integrator | *Dionaea* | 3 B/synapse | next |
| eligibility trace | e-prop, nerve nets | 4 B/synapse | next |
| flux-reinforced graph memory | *Physarum* | 8-16 B/edge | retrieval, routing |
| diffuse neuromodulator field | nerve nets, volume transmission | 64x64x16 floats = 0.25 MB | global state without a controller |
| distributed detector pool | immune system | 16 B/detector | novelty, anomaly |
| gap-junction relaxation | ctenophore syncytium | 8 B/edge | diffusion-like inference |
| per-word AoA + boundary F1 | child learning | none (measurement) | benchmarks |
| meta-d', ECE, post-error slowing | consciousness proxies | none (measurement) | benchmarks |

What to avoid (measured evidence against): dense weight matrices (10k^2 fp32 = 400 MB), backprop,
hardcoded central pattern generators, global all-reduce per step, float64.

## Part 5 · Measurement gaps in our own project

1. We report corpus-level top-1. Children's data predicts we should report **per-word AoA** and
   **boundary F1**.
2. We report anomaly AUC but not against a rolling null (NAB uses a rolling Gaussian; our AUC 0.711
   is not comparable to NAB's 65.3).
3. We have no metacognition measurement at all: no confidence calibration, no error monitoring.
4. We have no "does it still know yesterday's list" curve across *many* lists; retention is measured
   over one list only.

## Sources

Saffran, Aslin & Newport 1996 Science 274:1926; Gilkerson et al. 2017 AJSLP; Gilkerson et al. 2018
Pediatrics 141:e20173254; Brysbaert et al. 2016 Front Psychol 7:1116; Frank 2023 TICS; BabyLM 2025
(2504.08165); Chang & Bergen 2022 TACL; Frank 2026 (2608.17120); Goriely et al. 2025 (2504.03338);
Chang et al. 2024 (2408.03617); Medina et al. 2011 PNAS 108:9014; Mintz 2003 Cognition 90:91;
Nakagaki et al. 2000 Nature 407:470; Tero et al. 2010 Science 327:439; Bonifaci et al. 2012
(1106.0423); Reid et al. 2012 PNAS 109:17490; Bohm et al. 2016 Curr Biol 26:1286; Toyota et al.
2018 Science 361:1112; Gagliano et al. 2014 Oecologia 176:1063; Nath et al. 2017 Curr Biol;
Tripedalia conditioning 2023 Curr Biol; sea anemone conditioning 2023 PNAS; Karst et al. 2023 Nat
Ecol Evol 7:1439; Perelson & Oster 1979 J Theor Biol; Forrest et al. 1994; de Castro & Von Zumbusch
2002; Tononi 2004; Oizumi et al. 2014; Dehaene & Changeux 2011; Lau 2022; Graziano 2013; Cogitate
2023 (2312.06637) and 2025 Nature; Fleming & Lau 2014 Nat Rev Neurosci.