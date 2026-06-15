# Algorithm: Q-Learning with a Contrastive Local Learning Network (CLLN)

## Given

- CLLN with edges i ∈ {1,...,E} and trainable gate voltages U_G,i bounded by [GMN, GMX] = [1.0, 5.5]
- Input nodes encoding state S_t as voltages U_in
- Output nodes n ∈ {1,...,N_A}, one per action a ∈ {1,...,N_A}
- Reward function R(S,A) with Gaussian noise, provided by the environment
- Transition function S_{t+1} = QState(S_t,A_t)

### Hyperparameters


| Symbol    | Meaning                               | Code value             |
| --------- | ------------------------------------- | ---------------------- |
| α         | Learning rate                         | `ALF = 10`             |
| η         | Nudge factor                          | `ETA = 0.1`            |
| γ         | Discount factor                       | `GAM = 0.5`            |
| ε_0, ε_∞  | Epsilon-greedy schedule, linear decay | `EPS = 0.1 -> EPM = 0` |
| T         | Total training steps                  | `schedule.total_steps = 300000`; stored as `STP` |
| B         | Batch period for applying gradients   | `BTH = 50`             |
| R_reset   | Random-state reset period             | `RES = 5`              |
| `DGC`     | Per-edge gradient clip                | `DGC = 0.1`            |
| seed/init | Initial gate voltage distribution     | `EXP.network.params = 1.5 + randn(E)*0.1` in the experiment loop |


## Pseudocode

```text

initialize U_{G,i} ← 1.5 + N(0, 0.1), clipped to [GMN, GMX]
initialize accumulated update δU_{G,i} ← 0  for all i
sample initial state S_0 ~ Uniform{states}

for t = 0, 1, ..., T-1 do
    ─── Free state for S_t ───────────────────────────────────────────
    impose input voltages encoding S_t at input nodes
    equilibrate network (no clamps) → node voltages U^F(S_t)
    record per-edge voltage drops ΔU^F_i = U^F_{from(i)} − U^F_{to(i)}
    read free outputs F(S_t) = { U^{out,F}_n }_{n=1..N_A}

    ─── ε-greedy action selection ─────────────────────────────────────
    ε_t  ←  ε_0·(1 − t/T) + ε_∞·(t/T)
    if rand() < ε_t then
        A_t ← Uniform{1,...,N_A}                    (random action)
    else
        A_t ← argmax_n F(S_t)                       (greedy action)
    end if

    ─── Environment step ─────────────────────────────────────────────
    R_t      ← R(S_t, A_t) + N(0, σ_R^2)
    S_{t+1}  ← QState(S_t, A_t)
    append R_t to reward log

    ─── Bootstrap target (Bellman-like) using free pass of S_{t+1} ────
    impose S_{t+1}, equilibrate (free)  → F(S_{t+1})
    L_t ← R_t + γ · ( max_n F(S_{t+1})  −  mean_n F(S_{t+1}) )
        (subtracting the mean relaxes a constraint on the average
         output and improves performance for the small network)

    ─── Clamped state for S_t ────────────────────────────────────────
    build label vector L for the N_A output nodes:
        L_n  ← U^{out,F}_n(S_t)        for n ≠ A_t   (other actions held)
        L_{A_t} ← L_t                                (action's output)
    nudge each clamped output toward its label (η-clamping):
        U^{out,C}_n ← (1 − η)·U^{out,F}_n + η·L_n   for n = 1..N_A
    impose inputs of S_t AND U^{out,C}_n on outputs,
        equilibrate network → U^C(S_t)
    record per-edge clamped drops ΔU^C_i = U^C_{from(i)} − U^C_{to(i)}

    ─── Coupled-Learning contrastive update (eq. local-rule) ─────────
    for i = 1..E do
        δU_{G,i} ← δU_{G,i} + (α/η) · [ (ΔU^F_i)^2 − (ΔU^C_i)^2 ]
    end for
        (because ∂G_i/∂U_{G,i}=S, gradient on G_i ≡ gradient on U_{G,i})

    ─── Apply update in batches of B steps ────────────────────────────
    if (t+1) mod B == 0 then
        δU_{G,i} ← clip(δU_{G,i}, −DGC, +DGC)
        U_{G,i}  ← clip( U_{G,i} + δU_{G,i}, GMN, GMX )
        δU_{G,i} ← 0
    end if

    ─── State carry-over and periodic random reset ────────────────────
    S_t ← S_{t+1}
    if (t+1) mod R_reset == 0 then
        S_t ← Uniform{states}        (re-roll state for exploration)
    end if
end for

return { U_{G,i} }
```

---

## Mapping between the manuscript notation and the code

In the code references below, `self` is the `Experiment` object, and `self.network` is the contained `ResistorNetwork`.

| Manuscript quantity                                                    | Code symbol / location                                                                       |
| ---------------------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| Gate voltage U_G,i (learnable parameter)                               | `self.network.params` (clipped to `[self.GMN, self.GMX]`)                                    |
| Accumulated gate update δU_G,i                                         | `self.update`                                                                                |
| Input encoding of state S_t -> U_in                                    | `self.DATA[train_idx, :self.INP]`, imposed at `self.IONODE[:self.INP]`                       |
| Free-state node voltages U^F                                           | `self.freestates[train_idx]` (from `self.getState(..., free=True)`)                          |
| Free outputs F(S_t)                                                    | `self.freeOut(train_idx)`                                                                    |
| Clamped outputs U_out,C,n                                              | `self.getClamps(freeOut, labels)` with nudge `(1 - self.ETA)f + self.ETA*L`                  |
| Clamped-state node voltages U^C                                        | `self.clampstates[train_idx]`                                                               |
| Edge drops ΔU^F_i, ΔU^C_i                                              | `Vf`, `Vc` computed from `self.network.node_from / self.network.node_to`                     |
| Local rule δU_G,i = α[(ΔU^F_i)^2 - (ΔU^C_i)^2]                         | accumulated in `self.update`; core term is `self.ALF * (Vf**2 - Vc**2) / self.ETA`           |
| Nudge factor η                                                         | `self.ETA = 0.1`                                                                             |
| Learning rate α                                                        | `self.ALF = 10`                                                                              |
| Discount γ                                                             | `self.GAM = 0.5`                                                                             |
| Future-weighted target L_t = R_t + γ[max F(S_{t+1}) - mean F(S_{t+1})] | `reward + self.GAM*(max(self.freeOut(new_env_state)) - mean(self.freeOut(new_env_state)))`   |
| ε-greedy schedule (linear ε_0 -> ε_∞)                                  | `frac = t/self.STP; eps = self.EPM*frac + self.EPS*(1-frac)`                                 |
| Total training steps T                                                 | `self.STP = self.SCHEDULE.total_steps`                                                       |
| Batched updates every B steps                                          | schedule action `BATCH [every 50 steps]` -> `self.apply_batch()`                             |
| Periodic random state reset                                            | schedule action `RESET [every 5 steps]` -> re-roll `self.TDS` in `Navigation.ipynb`           |


