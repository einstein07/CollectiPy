# AUDIT — §3 seed-scheme battery

Scheme `frontier-v1`; trial identity (Δθ = 60°, δ_Q = 100 bp); 5 run_ids; env trace = shared percept draws, ticks ≤ 50; git `7d077fc56617315f0720bab084c3829b78248233-dirty`.

Routing (RECON §5): `sensory_stream.seed` ← `env_seed(…, 'sensory')`; arena `random_seed` ← `model_seed(model, …)` — the arena seed feeds only model-private generators in this simulator.

| check | result | detail |
|---|---|---|
| (a) determinism [ra] | **PASS** | bitwise-identical percept/position/noise logs |
| (a) determinism [ddm] | **PASS** | bitwise-identical percept/position/noise logs |
| (b) cross-model | **PASS** | RA≡DDM percept draws bitwise-identical at equal (tick, target) on all 5 run_ids (≥20 shared coordinates each) |
| (c) parameter invariance [ra] | **PASS** | (v=0.5, û=1.0) vs (v=0.2, û=0.65): identical env traces |
| (c) parameter invariance [ddm] | **PASS** | c_e=8 vs c_e=300: identical env traces |
| (d) stream separation [ra] env | **PASS** | env trace untouched by a model_seed change |
| (d) stream separation [ra] behavior | **PASS** | private-noise trajectory changed as expected |
| (d) stream separation [ddm] env | **PASS** | env trace untouched by a model_seed change |
| (d) stream separation [ddm] behavior | **PASS** | trajectory UNCHANGED — this model may consume no private noise in this configuration (informational, not a failure) |

Overall: **PASS** (13 s wall). Submission unblocked.
