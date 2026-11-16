# 37 · Human brain as an engineering spec (7-persona panel)

Seven agents ran in parallel: computational neuroanatomy, cortical microcircuits, hippocampal
physiology, basal ganglia/RL, sleep, dendritic computation, neuromorphic silicon. Numbers marked
"unverified" were reported from memory by an agent that could not reach the primary source; treat
them as leads.

## 1 · The hardware budget

| quantity | value | source |
|---|---|---|
| neurons | 86.1e9 (cortex 16.3e9, cerebellum 68.9e9) | Herculano-Houzel 2009, doi:10.1002/cne.21974 |
| glia | 84.6e9 (~1:1, not 10:1) | same |
| synapses | 1e14 - 1.5e14, ~7,000-9,000/neuron in cortex | (unverified) |
| power | 20 W, ~20% of BMR, 14 mW/g | Attwell & Laughlin 2001 |
| energy split | APs 21%, postsynaptic receptors 50% (revised); old split APs 47% | Howarth et al. 2012, doi:10.1038/jcbfm.2012.35 |
| ATP per AP | ~1.2e8 (mossy fibre 2.6e8, 4x cheaper) | Alle et al. 2009, doi:10.1126/science.1174331 |
| active fraction | 1-5% of neurons at any instant | (approximate) |
| firing rate | median ~1 Hz, log-normal | Roxin 2011 |
| spike timing | synaptic delay 0.5-1 ms, somatic AP 0.5-2 ms, max 200 Hz | (standard) |
| conduction | 0.3-10 m/s cortical axons, up to 120 m/s myelinated | (standard) |
| minicolumn | 80-100 neurons, ~1e5 neurons/mm3 | Buxhoeveden & Casanova 2002; Markram 2015 |

**What this means for a 65 MB system.** 65 MB over 1e14 synapses is 0.65 bytes/synapse, so the
cortex's fan-in (7,000-9,000) is unreachable: cap fan-in at 10-50 and recover depth with ~6 stages.
Our 4% SDR density already sits inside the biological 1-5%; raising it buys nothing. At a 1 Hz median
rate and 6.5e7 units, the touch budget is ~8 bytes per activation, which is exactly a 64-bit word
popcount. The "2.5 PB" memory estimate (Bartsch et al. 2011) is synapse-count times bits/synapse and
has no addressable read/write; do not use it as a target.

## 2 · Neocortex

- Wiring (Douglas & Martin 2004, doi:10.1146/annurev.neuro.27.070203.144152; Harris & Shepherd
  2015, doi:10.1038/nn.3917): TC -> L4 -> L2/3 -> L5B -> subcortical; L6 -> thalamus; L1 apical
  dendrites receive higher-area feedback and matrix thalamus.
- Predictive coding (Bastos et al. 2012, doi:10.1016/j.neuron.2012.10.038): feedforward in gamma
  30-80 Hz via L4, feedback in alpha-beta 8-30 Hz via L6/L1. The often-repeated "10x more feedback
  than feedforward" is not in the abstract (unverified).
- Measured prediction-error signatures: MMN 100-250 ms, 1-2 uV; repetition suppression BOLD -20 to
  -40%; stimulus-specific adaptation -50 to -70% spikes (approximate).
- HTM (Hawkins & Ahmad 2016, doi:10.3389/fncir.2016.00023): 320k transitions learned, ~300
  patterns/cell, hundreds of contexts; burst = column active with no prediction, predict = >=1 cell
  in predictive state.
- Dendrites: human L2/3 dCaAP is graded, threshold about -36 mV, 43.8 mV peak, 3 ms rise
  (Gidon et al. 2020, doi:10.1126/science.aax6239); a single cell solves XOR (Polsky et al. 2004,
  doi:10.1038/nn1253). NMDA plateau needs a 20-40 um cluster, lasts hundreds of ms, proximal inputs
  ~7x more effective (Major et al. 2008, doi:10.1152/jn.00011.2008).

Cost of a faithful version: SDR 2048x41 = 82 B/pattern; 320k transitions at 8 B = 2.6 MB; dendrites
16 segments x 40 synapses x 1 B = 656 B/cell, ~99k cells in 65 MB. Total ~24 MB. Do not copy the
cat-V1 weight sets, gamma synchrony as a clock, or any backprop.

## 3 · Hippocampus

- Place fields 10-30 cm in 1-2 m arenas; global remapping on context change; grid modules with
  spacing ratios 1.4-1.7 (Stensola et al. 2012, doi:10.1126/science.1219801); head-direction cells
  ~10% of MEC, 60-80 deg tuning.
- Replay: waking 100-200 ms sequences compressed 6-10x into ~20 ms during SWS (Wilson & McNaughton
  1994, doi:10.1126/science.8034517); ripples 140-200 Hz; reverse replay is reward-modulated (Singer
  & Frank 2009, doi:10.1016/j.neuron.2009.11.016; Ambrose et al. 2016, doi:10.1016/j.neuron.2016.07.047).
- Indexing: cortex stores content, hippocampus stores a sparse index (Teyler & DiScenna 1986,
  doi:10.1037/0735-7044.100.2.147; Teyler & Rudy 2007, doi:10.1002/hipo.20331). Inferential, not a
  circuit proof.
- Capacity: Hopfield 0.14N (doi:10.1073/pnas.79.8.2554); CA3 2.5e5 cells -> ~3.5e4 ideal, 1e4-1e5
  practical; dentate gyrus expands ~5x; CA3/CA1 sparse at ~2-5%.
- One-shot: novelty-triggered dopamine and noradrenaline enable single-trial CA1 encoding; sleep
  consolidation gains ~10-20% (task-dependent, approximate).

Cheap replay for us: store compressed sequences as delta-encoded unit-ID lists (~20 ms, 6-10x), then
re-stimulate the recurrent layer with the list time-scaled. No gradients, no recompute.

## 4 · Basal ganglia and dopamine

- Schultz, Dayan & Montague 1997 (doi:10.1126/science.275.5306.1593): tonic 1-8 Hz (mean 4-5),
  phasic bursts 20-100 Hz (peaks ~120), omission dips to 1-3 Hz, latency 100-150 ms; ~75% of DA
  neurons respond to unpredicted reward (Mirenowicz & Schultz 1994, doi:10.1152/jn.1994.72.2.1024).
- Sufficiency: VTA bursts at reward unblock learning (Steinberg et al. 2013, doi:10.1038/nn.3413);
  DA stimulation alone drives CPP (Tsai et al. 2009, doi:10.1126/science.1169463).
- Three-factor rule: corticostriatal HFS alone gives LTD, HFS plus DA gives LTP (Reynolds, Hyland &
  Wickens 2001, doi:10.1038/35092560); pre-post coincidence ~100 ms, DA within 100-300 ms. D2 SPNs
  detect DA dips as short as 0.4 s (Iino et al. 2020, doi:10.1038/s41586-020-2115-1).
- Receptor affinities make the scalar story work: D1 Kd ~1 uM (phasic detector), D2 Kd ~10-30 nM
  (tonic/dip detector); phasic DA 1-10 uM vs ambient 20-50 nM. DA is heterogeneous (Engelhard et al.
  2019, doi:10.1038/s41586-019-1261-7), so "one scalar" is a first approximation.

Our design already matches this: one scalar delta broadcast, per-action value heads, eligibility trace
with tau ~1 s, fp16 traces to halve weight memory.

## 5 · Sleep

- Cycle 70-120 min, 4-6/night; N1 5%, N2 45-55%, N3 15-25%, REM 20-25%; spindles 11-15 Hz at
  ~1-2/min in N2 (detection-dependent); slow oscillation <1 Hz; ripples 140-200 Hz.
- SHY: 6,920 mouse synapses, axon-spine interface down ~18% after sleep vs wake, proportional to
  synapse size, large synapses spared (de Vivo et al. 2017, doi:10.1126/science.aah5982; Tononi &
  Cirelli 2014, doi:10.1016/j.neuron.2013.11.010). This is the mechanism that keeps a Hebbian matrix
  from saturating, and it is exactly the term our matrix lacks.
- Targeted memory reactivation meta-analysis: 91 experiments, 212 effect sizes, N=2,004, Hedges
  g=0.29 [0.21, 0.38]; N2 g=0.32, SWS g=0.27, no reliable REM or wake effect (Hu et al. 2020,
  doi:10.1037/bul0000223).

Engineering: replay the top-k episodes by reward/novelty, forward for prediction and reverse for
credit, then downscale the weakest 80% of synapses by ~0.82 and renormalise. 65 MB read+write is
~23 ms at 5 GB/s, so 100 consolidation cycles cost ~2.3 s.

## 6 · Neuromorphic silicon: where a CPU is actually fine

| platform | key number | source |
|---|---|---|
| Loihi | 128 neurocores, ~130k neurons/chip, LCA ~3 orders better EDP | doi:10.1109/MM.2018.112130359 |
| Loihi KWS | 0.08-0.11 W, 4 ms vs 148 ms CPU (~37x) | arXiv:1812.01739 |
| Loihi 2 PilotNet | 0.05 mJ vs Jetson 7.6 mJ (152x) | arXiv:2310.03251 |
| SpiNNaker | 0.60 uJ per synaptic event vs NEST 5.8 uJ | arXiv:1909.08665 |
| BrainScaleS-2 | 1000x accelerated, 5.6 W, 276 us/sample | doi:10.3389/fnins.2022.795876 |
| TrueNorth | 1M neurons, 256M synapses, 63 mW at 400x240@30fps | doi:10.1126/science.1254642 |
| Akida AKD1000 | ~45 uJ/inference, ~97% below Xeon | arXiv:2408.03336 |

Roofline for us: peak RSS 11-15 MB; a 65 MB stream is 3.25 ms at 20 GB/s and ~0.65 mJ at 10 pJ/B.
Arithmetic intensity 0.25 MAC/B is ~10x below the 2.4 FLOP/B ridge, so we are bandwidth-bound, not
compute-bound. A CPU is not categorically 1000x behind: Loihi wins mainly through sparsity plus
on-chip weights (at 2% active, ~27x). Verdict: stay on CPU, use bit-parallel popcount and blocked
int8 sparse mat-vec; hardware is only needed for large scattered weight sets with sub-pJ/SOP.

## 7 · The three changes this implies for CortexMesh

1. **Sleep downscaling** (SHY). Every N tokens, scale the weakest 80% of synapses by 0.82 and
   renormalise. This is the only mechanism in the whole panel that directly attacks the saturation
   we measured in D41-D44, and it has a measured biological effect size (-18%).
2. **Eligibility traces + scalar delta** (basal ganglia). tau ~1 s, fp16, 4 B/synapse. Gives delayed
   credit assignment without backprop.
3. **Dendritic segments** (2-level readout). 16 segments x 40 synapses x 1 B per cell instead of a
   dense row, which is what makes a linear readout nonlinear (XOR) at ~656 B/cell.

Everything else in the panel is measurement, not architecture: per-word AoA curves, boundary F1,
meta-d', ECE. Those belong in the benchmarks, not the model.