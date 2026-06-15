# Algorithm: Q-Learning with a Contrastive Local Learning Network (CLLN)

-------------------------------------------------------------------
Given:
    - CLLN with edges i ∈ {1,...,E}, trainable gate voltages U_{G,i}
      (bounded U_{G,i} ∈ [GMN, GMX] = [1.0, 5.5])
    - Input nodes (encode state S_t as voltages U^in)
    - Output nodes n ∈ {1,...,N_A}, one per action a ∈ {1,...,N_A}
    - Reward function R(S, A) (+ Gaussian noise) provided by environment
    - Transition function S_{t+1} = QState(S_t, A_t)
    - Hyperparameters:
        α            learning rate                          (ALF = 10)
        η            nudge factor                           (ETA = 0.1)
        γ            discount factor                        (GAM = 0.5)
        ε_0, ε_∞     ε-greedy schedule (linear decay)       (EPS=0.1 → EPM=0)
        T            total training steps                   (STP = 3·10^5)
        B            batch period (gradient applied)        (BTH = 50)
        R_reset      random-state reset period              (RES = 5)
        DGC          per-edge gradient clip                 (DGC = 0.1)
        seed/init    U_{G,i} ~ N(1.5, 0.1)

1: initialize U_{G,i} ← N(1.5, 0.1), clipped to [GMN, GMX]
2: initialize accumulated update δU_{G,i} ← 0  for all i
3: sample initial state S_0 ~ Uniform{states}

4: for t = 0, 1, ..., T-1 do
5:     ─── Free state for S_t ───────────────────────────────────────────
6:     impose input voltages encoding S_t at input nodes
7:     equilibrate network (no clamps) → node voltages U^F(S_t)
8:     record per-edge voltage drops ΔU^F_i = U^F_{from(i)} − U^F_{to(i)}
9:     read free outputs F(S_t) = { U^{out,F}_n }_{n=1..N_A}
10:
11:    ─── ε-greedy action selection ─────────────────────────────────────
12:    ε_t  ←  ε_0·(1 − t/T) + ε_∞·(t/T)
13:    if rand() < ε_t then
14:        A_t ← Uniform{1,...,N_A}                    (random action)
15:    else
16:        A_t ← argmax_n F(S_t)                       (greedy action)
17:    end if
18:
19:    ─── Environment step ─────────────────────────────────────────────
20:    R_t      ← R(S_t, A_t) + N(0, σ_R^2)
21:    S_{t+1}  ← QState(S_t, A_t)
22:    append R_t to reward log
23:
24:    ─── Bootstrap target (Bellman-like) using free pass of S_{t+1} ────
25:    impose S_{t+1}, equilibrate (free)  → F(S_{t+1})
26:    L_t ← R_t + γ · ( max_n F(S_{t+1})  −  mean_n F(S_{t+1}) )
27:        (subtracting the mean relaxes a constraint on the average
28:         output and improves performance for the small network)
29:
30:    ─── Clamped state for S_t ────────────────────────────────────────
31:    build label vector L for the N_A output nodes:
32:        L_n  ← U^{out,F}_n(S_t)        for n ≠ A_t   (other actions held)
33:        L_{A_t} ← L_t                                (action's output)
34:    nudge each clamped output toward its label (η-clamping):
35:        U^{out,C}_n ← (1 − η)·U^{out,F}_n + η·L_n   for n = 1..N_A
36:    impose inputs of S_t AND U^{out,C}_n on outputs,
37:        equilibrate network → U^C(S_t)
38:    record per-edge clamped drops ΔU^C_i = U^C_{from(i)} − U^C_{to(i)}
39:
40:    ─── Coupled-Learning contrastive update (eq. local-rule) ─────────
41:    for i = 1..E do
42:        δU_{G,i} ← δU_{G,i} + (α/η) · [ (ΔU^F_i)^2 − (ΔU^C_i)^2 ]
43:    end for
44:        (because ∂G_i/∂U_{G,i}=S, gradient on G_i ≡ gradient on U_{G,i})
45:
46:    ─── Apply update in batches of B steps ────────────────────────────
47:    if (t+1) mod B == 0 then
48:        δU_{G,i} ← clip(δU_{G,i}, −DGC, +DGC)
49:        U_{G,i}  ← clip( U_{G,i} + δU_{G,i}, GMN, GMX )
50:        δU_{G,i} ← 0
51:    end if
52:
53:    ─── State carry-over and periodic random reset ────────────────────
54:    S_t ← S_{t+1}
55:    if (t+1) mod R_reset == 0 then
56:        S_t ← Uniform{states}        (re-roll state for exploration)
57:    end if
58: end for

59: return { U_{G,i} }
```

---

## Mapping between the manuscript notation and the code

| Manuscript quantity                                 | Code symbol / location                                                 |
|-----------------------------------------------------|-------------------------------------------------------------------------|
| Gate voltage $U_{G,i}$ (learnable parameter)        | `network.params` (clipped to `[GMN, GMX]`)                              |
| Input encoding of state $S_t$ → $U^{\mathrm{in}}$    | `DATA[train_idx, :INP]` imposed via `solve_equilibrium`                |
| Free-state node voltages $U^{F}$                    | `freestates[train_idx]` (from `getState(..., free=True)`)              |
| Free outputs $\mathcal{F}(S_t)$                     | `freeOut(train_idx)`                                                    |
| Clamped outputs $U^{\mathrm{out},C}_n$              | `getClamps(freeOut, labels)` with nudge $(1-\eta)f + \eta L$           |
| Clamped-state node voltages $U^{C}$                 | `clampstates[train_idx]`                                                |
| Edge drops $\Delta U^{F}_i$, $\Delta U^{C}_i$        | `Vf`, `Vc` (computed from `node_from / node_to`)                       |
| Local rule $\delta U_{G,i} = \alpha[(\Delta U^F_i)^2 - (\Delta U^C_i)^2]$ | `self.update += ... * ALF * (Vf**2 - Vc**2) / ETA` |
| Nudge factor $\eta$                                 | `ETA = 0.1`                                                             |
| Learning rate $\alpha$                              | `ALF = 10`                                                              |
| Discount $\gamma$                                   | `GAM = 0.5`                                                             |
| Future-weighted target $L_t = R_t + \gamma[\max\mathcal{F}(S_{t+1}) - \mathrm{mean}\,\mathcal{F}(S_{t+1})]$ | `reward + GAM*(max(freeOut(new_env_state)) - mean(freeOut(new_env_state)))` |
| ε-greedy schedule (linear $\epsilon_0\!\to\!\epsilon_\infty$) | `frac = t/STP; eps = EPM*frac + EPS*(1-frac)`                  |
| Batched updates every $B$ steps                     | action list `['BATCH', -BTH]` → `apply_batch()` (clip by `DGC`, clip params to `[GMN,GMX]`) |
| Periodic random state reset                         | action list `['RESET', -RES]` → re-roll `self.TDS`                     |

## Differences vs. the manuscript description

- The manuscript example uses **4 states / 4 actions** with one-hot-like
  voltage encodings $[1\,0\,1\,0]$, $[0\,1\,0\,1]$, $[1\,1\,0\,0]$, $[0\,0\,1\,1]$ V
  and the trivial environment `next_state = action`. This notebook generalises
  to a **3×3 grid (9 states) with 5 actions** (`↑ ↓ ← → o`); the state is
  encoded as two analog input voltages giving (row, column) and an additional
  fixed input node, and the environment moves on the grid (with walls).
- The reward landscape adds a small **shaping** term that decays with Manhattan
  distance from `actual_target` (`shaping = 1e-5`); reward noise scale
  `_NOISE_SCALE` is set to 0 in this run.
- The "subtract the mean" term in $L_t$ is implemented exactly as written in
  the manuscript: `GAM*(max(F(S_{t+1})) - mean(F(S_{t+1})))`.
- Updates are accumulated over `BTH = 50` steps before being applied (matches
  "Updates are batched and imposed every 50 steps").
- An additional periodic random **state reset every `RES = 5` steps** is used
  to keep exploration broad (not stated in the manuscript snippet but present
  in the code).

## Differences vs. Algorithm 1 (MPO with Mirrored Data)

This procedure shares the high-level "sample → score → improve parametric
policy" loop, but the score and the policy update are very different:

| Algorithm 1 (MPO)                                                                 | CLLN Q-learning (this notebook)                                       |
|-----------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| Sample $N$ actions $a_i \sim \pi_k(\cdot|s_j)$ per state                          | One action $A_t$ via ε-greedy on free outputs of the network          |
| Score with critic $Q_{ij} = \hat Q(s_j, a_i)$ and weight $q_{ij}\propto e^{Q_{ij}/\tau}$ | Score with one-step Bellman target $L_t = R_t + \gamma[\max \mathcal{F}(S_{t+1}) - \mathrm{mean}\,\mathcal{F}(S_{t+1})]$ |
| Mirror states and actions to build augmented dataset                              | No data mirroring                                                      |
| Update $\pi_\theta$ by weighted maximum-likelihood + KL regulariser               | Update gate voltages by contrastive local rule (Coupled Learning)     |
| Replay buffer over many transitions                                               | Online; batched accumulation over $B=50$ consecutive transitions      |
```

