# Roadmap: learning RL on this project

A personal learning plan on top of the original ping-pong balancing project. Three changes, in this order:

1. [Move the wrist camera and retrain perception](#1-move-the-camera)
2. [Swap PPO for GRPO](#2-ppo--grpo)
3. [New task: a tray carrying a glass of water](#3-new-task-tray--glass-of-water)

Each step changes one thing, so when a result is worse you know why.

## Status

| | |
|---|---|
| MuJoCo | upgraded 3.17 → **4.0.0** (`pyproject.toml`). No code changes needed. |
| Pretrained controller on 4.0 | 195/200 (97.5%) with training randomization, same as on 3.17 (README) |
| Full camera pipeline on 4.0 | 30/30 (10 episodes × latency 0, 2, 4) |
| Viewer | new `--seconds S` flag for longer episodes (default 10 s = the training length). A 20 s run with kicks every 4 s failed at ~11 s: the policy was only trained on 10 s episodes. |

### Evaluation tools

| Command | What it measures |
|---|---|
| `uv run python -m sim.stress_eval` | success rate vs kick strength (0–0.25 m/s, one kick per 4 s) × episode length (10/20/30 s). Writes `outputs/stress/`. |
| `uv run python -m sim.critic_plot` | whether PPO's critic value predicts a drop, and how early. Writes `outputs/critic/`. |
| `uv run pytest` | tests for the shared episode runner (`sim/stress.py`) and the alarm metrics |

Baseline (pretrained PPO controller, MuJoCo 4.0):
- No kicks: 100% at 10, 20 and 30 s, so episode length alone is not a problem.
- Breaking point at 0.15–0.2 m/s kicks: ~80% → ~50% success at 10 s. At 0.25 m/s, 0–5% beyond 10 s.
  It was never kicked in training: the only ball motion was a start velocity of at most 0.03 m/s per axis.
- The critic is a weak drop predictor: its value falls only ~0.2–0.5 s before a drop, and balanced episodes dip just as
  low right after a kick (at the 1st-percentile threshold: catches 18% of drops, 56% false alarms).

### Kicks in training

`BalanceEnv` can kick the ball mid-episode (`kick_min`/`kick_max` m/s, one per `kick_every` s, none in the first
`kick_grace` s or the last `kick_margin` s). The schedule comes from the reset seed only, so episodes reset with the same
seed get identical kicks, which is what GRPO groups need. The policy is not told about kicks. Training flags:
`--episode-s`, `--kick-min`, `--kick-max`, `--kick-every`, `--kick-margin`. The stress tools use the same code.

A kick sets the ball sliding without spin; friction turns that into rolling within one step, so a hollow ping-pong ball
(I = ⅔mr²) ends up moving at ⅗ of the kick speed.

### PPO with kicks: results

PPO from scratch, 4 s episodes, one kick of 0–0.3 m/s, otherwise the README settings. Stress eval at 10 s
(jitter = mean |a_t − a_t−1|; the original controller is ~0.03, visible shaking ~0.3):

| smooth penalty | steps | jitter | no kick | 0.2 m/s | 0.3 m/s | 0.4 m/s |
|---|---|---|---|---|---|---|
| original controller | 28.7M | ~0.03 | 100% | 52% | 0% | 0% |
| 0.01 (old default) | 2.0M (best) | 0.19 | 100% | 95% | 65% | 25% |
| 0.01 | 6.3M (last) | 0.09 | 92% | 90% | 55% | 2% |
| 0.1 | 2.15M | 0.05 | 100% | 90% | 65% | 22% |
| **0.3** | **2.15M** | **0.03** | **100%** | **95%** | **70%** | **25%** |
| 1.0 | 2.15M | 0.02 | 100% | 85% | 50% | 15% |

Lessons:
- Kick training moves the breaking point from ~0.2 to ~0.3 m/s in ~2M steps, and it holds on 30 s episodes.
- With the old 0.01 smoothness penalty the policy learned a visible tremor: shaking was almost free in the reward, and
  soft servos hid it in most training episodes. It only shows with stiff servos and low latency.
- 0.3 removes the tremor at no cost in robustness; 1.0 is too sluggish for strong kicks.
- Training success (~95% throughout) hid both the tremor and the 6.3M checkpoint's regression: always run the stress eval.
- Set `--iters` to the real run length so the learning rate decays over it.

Chosen PPO baseline: penalty 0.3, 2.15M steps (kept locally, not in git: `outputs/weights/ppo_kicks_smooth.pt`).

### GRPO training

`sim/grpo.py` (advantages + clipped update) and `sim/train_grpo.py` (loop), with `EpisodicVecEnv` in `sim/vec_env.py`
(reset with given seeds, no auto-reset). Each iteration: P seeds x G episodes, run to the end; advantage per episode =
(return - group mean) / group std, applied to every step. Same network class as PPO (critic never trained), same env
flags as `train_controller.py`, so checkpoints work with every eval tool. Comparison run, same conditions as PPO:

```bash
# PPO baseline with kicks
uv run python -m sim.train_controller --name ppo_kicks --iters 700 --smooth-penalty 0.3 --act-hist 3 --grip-rot -90 --max-latency 4 \
    --home-penalty 0.1 --servo-rand --ball-noise 1.0 --episode-s 4 --kick-min 0 --kick-max 0.3 --kick-every 2 --kick-margin 1
# GRPO, same conditions and step budget
uv run python -m sim.train_grpo --name grpo_kicks --total-steps 6.45e6 --smooth-penalty 0.3 --act-hist 3 --grip-rot -90 --max-latency 4 \
    --home-penalty 0.1 --servo-rand --ball-noise 1.0 --episode-s 4 --kick-min 0 --kick-max 0.3 --kick-every 2 --kick-margin 1
```

### GRPO vs PPO: results

Same conditions (4 s episodes, one 0–0.3 m/s kick, smooth penalty 0.3), G = 8 episodes per seed, 6 seeds per iteration.

| training success | 0.25M | 0.5M | 1M | 2.15M | 4M | 6.4M |
|---|---|---|---|---|---|---|
| PPO | 82% | 93% | 94% | 96% (end) | — | — |
| GRPO | 26% | 93% | 85% | 79% | 94% | 97% |

| stress eval, 10 s (30 s) | no kick | 0.2 m/s | 0.3 m/s | 0.4 m/s | jitter | time |
|---|---|---|---|---|---|---|
| PPO, 2.15M steps | 100% (100%) | 95% (85%) | 70% (38%) | 25% (2%) | 0.032 | 6 min |
| GRPO, 6.45M steps | 98% (95%) | 90% (80%) | 75% (60%) | 40% (10%) | 0.044 | 12 min |

- GRPO starts slower and is noisier (one score per episode vs PPO's per-step critic), but kept improving and ends
  better on strong kicks: all G attempts face the identical kick, so the comparison isolates the reaction to it.
- Every group had a learning signal (0% groups with identical returns).
- Open: PPO with the same 6.45M budget, and 2–3 training seeds per algorithm, to separate algorithm from budget and luck.

## How the pieces fit (what each change touches)

```
sim state ──(+ noise imitating the camera)──> controller obs ──> PPO policy ──> 5 joint deltas @ 30 Hz
wrist camera ──> detector ──> estimator ──> ball position      (only at eval / on the real arm)
```

The controller is trained on the simulator's ball state plus noise, **never on images**. The camera only matters
for `perception/` (the detector and estimator).

| Change | Files | Retrain |
|---|---|---|
| Camera position | `assets/so101/so101_new_calib_paddle.xml`, maybe `perception/` | estimator only |
| PPO → GRPO | new `sim/grpo.py`, `sim/train_controller.py` | controller |
| Tray + glass | new MJCF, new env, reward | controller (+ perception, optional) |

## 1. Move the camera

The smallest change and a good first loop of "edit → check → retrain → evaluate".

1. Edit the camera in `assets/so101/so101_new_calib_paddle.xml` (`<camera name="wrist" pos=... xyaxes=... fovy="62">`).
   The current pose was hand-tuned to match the real camera.
2. Render frames from the old and new views side by side. Check that the ball stays in view and the colour detector
   (`perception/detector.py`) still finds it. The current setup counts on the paddle being edge-on to the camera,
   so the ball stays near one image column. A different angle may change what the estimator needs as input.
3. Retrain the estimator: `uv run python -m perception.estimator --name my_estimator --frames 500000`.
   This also replaces the shipped estimator, which was trained on MuJoCo 3.17 renders.
4. Evaluate the full pipeline:
   `uv run python -m perception.estimator --eval weights/my_estimator.pt --latency 0,2,4 --workers 4 --servo-rand --episodes 40`
5. Leave the controller as it is. On the real arm, the physical camera has to move to the same pose.

**Done when:** the full-pipeline success rate is close to the current one (113/120 in the README).

### Ego camera: results

A base-mounted camera on a 65 cm post (`ego`, 62° lens, 256x192), kept as a second option next to the wrist camera.
Full pipeline (camera → detector → estimator → FdVel → `ppo_kicks_smooth` controller), randomized servos,
latency 0/2/4, 30 episodes each; "nudges" = two 0.25 m/s pushes per episode.

| estimator (`--camera`) | inputs | no nudges | 2 nudges | est err u/w (cm, no nudges) |
|---|---|---|---|---|
| wrist (shipped `weights/estimator.pt`) | 3 detections | 81/90 | 69/90 | 0.39 / 1.29 |
| ego, first version | + current joints | 78/90 | 25/90 | 1.02 / 1.32 |
| ego + joint history | + joints of all 3 frames | 86/90 | 23/90 | 0.91 / 1.49 |
| **ego**, decorrelated data | + joints of all 3 frames | **87/90** | 64/90 | 0.21 / 0.72 |
| ego_pose, decorrelated | + camera mount offset | 86/90 | 64/90 | 0.40 / 0.82 |
| ego_track, decorrelated | tracking camera, + pan/tilt | 83/90 | **68/90** | 0.21 / 0.62 |

What mattered, in order:

1. **The arm pose gave the ball away.** The data was collected with the controller acting on the true ball, so it tilts
   toward the ball: from the joint angles alone a quadratic fit predicts the ball's w with R² 0.69. An estimator that
   sees the joints learns that shortcut. In closed loop the arm follows the *estimate* instead, the shortcut is wrong,
   and the estimate drifts slowly (±1.4 cm within an episode, ~0.1 Hz) while the controller holds it at the centre.
   Fix: `--target-offset 0.3` — during collection the controller balances around a random moving target (per-episode
   offset + ~2 s drift), so the pose no longer tells where the ball is (R² 0.21). Drift 1.2/1.5 → 0.28/0.43 cm.
   Same idea as DAgger: train on the state distribution the estimator will meet in closed loop.
2. Joint history (each detection paired with the pose it was taken at): needed for a camera that does not move with
   the paddle, but on its own it barely changed anything (error at the fastest arm motion 0.81 → 0.76 cm).
3. Camera mount offset as input (`ego_pose`, from a calibration on the real arm): no gain. With the camera
   randomization off the remaining bias halves (w 0.60 → 0.33 cm), but the estimator did not learn to use the inputs.
4. Tracking camera (`ego_track`, `BalanceEnv(cam_track=True)`): pan/tilt servo steered by its own detections
   (`env.track_detection`, nominal FOV only, 6 rad/s limit), keeps the ball within ±0.02 of the image centre. About the
   same success as the fixed camera, but the error stays flat with ball distance (5-8 cm: 0.36/0.74 vs 0.67/1.01 cm),
   which is what a bouncing ball needs. On the real arm: two extra servos on the post.

Two sim pitfalls found on the way: a tracker aimed from the true ball position leaks the answer through its pan/tilt
(estimate error 0.06 cm, not achievable), and a camera moved by editing `m.cam_quat` renders one step late unless
`mj_camlight` runs after the change.

```bash
uv run python -m perception.estimator --name ego_dc --camera ego --target-offset 0.3 --controller outputs/weights/ppo_kicks_smooth.pt
uv run python -m perception.estimator --eval outputs/estimator/ego_dc.pt --controller outputs/weights/ppo_kicks_smooth.pt \
    --latency 0,2,4 --episodes 30 --servo-rand --nudges 2
```

### One policy for the arm and the camera: results

`sim/cam_policy_env.py`: one PPO policy with 7 actions (5 joint deltas + pan/tilt rates through the same latency)
that sees the ball only through the camera: last 3 tracked detections, pan/tilt of those frames, joints, past actions.
Asymmetric actor-critic: the critic also gets the true ball state, the true projection, the camera mount offset and the
latency (training only). No estimator, no centring rule: the policy decides where to look.

Training cannot render (PPO needs millions of steps), so detections are **simulated**: the ball is projected through
the camera (`sim/cam_model.py`, true randomized pose and FOV) and given detector-like errors fitted against the
renderer + detector (`sim/fit_cam_model.py`, 60k frames, the camera drifting off the ball to cover the image).
Held out: detection rate 96.5 vs 96.5%, x error p50/p99 0.004/0.013 vs 0.004/0.011, y 0.008/0.022 vs 0.003/0.020
(image units, -1..1). Simulated and rendered detections go through the same tracker
(`perception.detector.BallDetector(track=True)`). Getting the fit to match needed three detector-side fixes, found
by comparing the two:

1. A ball cut by the image border is a sliver whose centre and size are wrong; the detector reported it.
   `candidates()` keeps only whole balls: round (aspect <= 1.5, fills >= 55% of its circle), not touching the border.
2. A small orange part at the arm base (~3 px radius vs the ball's 5-7.5 px): min area 60 px.
3. The camera could tilt down at its own post (orange in some randomized episodes). Servo limits
   `CAM_AIM_LIMIT` = ±0.6 rad pan, ±0.5 rad tilt around the nominal aim (the paddle is ~0.33 rad below it).
   After these, detections with the ball out of view: 53% → 8.5% → 0%.

The tracker (`BallTracker`) predicts the ball's image position (constant velocity + the shift of the camera's own
turn, from its servos), keeps the candidate with the best box overlap (IoU), and coasts on the prediction
(visible = 0) for up to 5 frames.

Trained with the `ppo_kicks_smooth` recipe (latency 0-4, servo randomization, kicks up to 0.3 m/s every 2 s, 4 s
episodes, smooth penalty 0.3), from scratch, stopped at 2000 iterations (6.1M steps, ~35 min); best checkpoint at
1900 (96% training success). Same test as the table above (latency 0/2/4, 30 episodes each, 10 s):

| | no nudges | 2 nudges | latency 4, 2 nudges |
|---|---|---|---|
| wrist estimator + controller | 81/90 | 69/90 | 21/30 |
| ego_dc estimator + controller | 87/90 | 64/90 | 12/30 |
| ego_track (rule-steered camera) + estimator + controller | 83/90 | 68/90 | 15/30 |
| **arm + camera policy, rendered** | **90/90** | **69/90** | **19/30** |
| arm + camera policy, simulated detections | 88/90 | 62/90 | 14/30 |

- Best camera result without nudges (90/90), level with the best with nudges; the weak spot is still latency 4 with
  pushes: 19/30, better than the ego estimators (12-15/30) and close to the wrist camera (21/30; 30 episodes per
  cell, so ±2-3 is noise).
- Rendered evaluation is slightly *better* than simulated: the trained-on model misses 4.2% of in-view frames (the
  rendered evaluation lost the ball in 0.6% of frames) and scales errors up to 1.5x per episode, so it errs on the
  pessimistic side.
- It does not centre the ball: the ball sits ~0.35 (image units) off centre and the camera moves little
  (|pan/tilt command| ~0.1). Keeping the ball in view (99.4% of frames) is all balancing needs; looking ahead should
  matter for bouncing.

```bash
uv run python -m sim.fit_cam_model --frames 60000                       # fit the detection model
uv run python -m sim.train_controller --name cam_policy --cam-policy --act-hist 3 --grip-rot -90 --max-latency 4 \
    --servo-rand --home-penalty 0.1 --smooth-penalty 0.3 --episode-s 4 --kick-max 0.3 --kick-every 2 --iters 2000
uv run python -m sim.eval_cam_policy --policy outputs/controller/cam_policy/best.pt --latency 0,2,4 --episodes 30 --nudges 2
uv run mjpython -m sim.view_cam_policy --nudge 2.5                     # watch it
```

### Bouncing: kicks in 3 axes (iteration 1)

The ball can now move in all 3 axes: kicks get a random upward component, the ball flies off the paddle and bounces
when it lands, and the arm + camera policy is fine-tuned to catch it, settle it and keep it centred.

- Kicks (`--kick-z-min/--kick-z-max`, m/s): an upward speed on the kicks of a `--kick-z-frac` share of the episodes;
  the rest keep the original sideways kicks, so plain balancing stays trained.
- Elastic contact (`--contact-damp lo,hi`, MuJoCo direct `solref = (-40000, -damping)`): drop tests on a still paddle
  give restitution ~0.86 (damping 20), ~0.80 (30), ~0.75 (40); the original contact lands dead (~0.1).
- Termination: the ball may be in the air; the episode fails if it falls past the blade, lands off it, goes > 20 cm
  sideways or > 50 cm up. Success also needs it back on the blade at the end.
- **Reward unchanged** (centred, slow ball, smooth, near home, -5 on a drop): the best answer to an upward kick is to
  follow the ball, cushion the landing and settle it. Nothing rewards keeping it bouncing.
- Starts from the exact `cam_policy` weights: the actor's inputs are unchanged; the critic gets 2 more (ball lift and
  normal velocity) with zero initial weights (`expand_critic`, `--init-from`).

**Forgetting.** The first fine-tune (exploration std raised 0.09 -> 0.15, 40% sideways episodes) lost balancing
faster than it learned bouncing: after 400 iterations, sideways pushes 60 -> 34/90, upward kicks 19 -> 26/90
(`outputs/controller/cam_bounce_v1_forgot`). Restarted with `--critic-warmup 100` (only the critic learns for 100
iterations, so its value estimates fit the new task before they move the policy), the base std, and 50% sideways
episodes: sideways training success then stayed at the base level (76%) while upward kicks improved.

Result (`cam_bounce`, 1500 iterations from `cam_policy`, 4.6M steps, 25 min; upward kicks 0.5-1.5 m/s in 50% of the
episodes, elastic contact damping 20-40). Rendered, latency 0/2/4, 30 episodes each, two pushes per episode:

| test | `cam_policy` (balance only) | `cam_bounce` |
|---|---|---|
| pushes with an upward kick 0.5-1.5 m/s, elastic contact | 21/90 (latency 0/2/4: 15/6/0) | **31/90** (15/14/2) |
| sideways pushes, original contact (the #2 test) | 69/90 | 67/90 |
| sideways pushes, elastic contact | 60/90 (simulated detections) | 64/90 |
| no pushes, elastic contact | 90/90 (original contact) | 88/90 |

- Balancing is kept (within the ±2-3 noise of 30 episodes per cell); upward kicks +10, mostly at latency 2 (6 -> 14).
- Most upward kicks are still lost, almost all of them at latency 4: catching a bouncing ball needs a faster
  reaction than settling a rolling one. Training success on upward kicks stayed at 25-40% from iteration 500 on.
- Command: as in the PR (`train_controller --cam-policy --init-from outputs/controller/cam_policy/best.pt
  --critic-warmup 100 --lr 1e-4 --episode-s 5 --kick-margin 1.5 --kick-z-min 0.5 --kick-z-max 1.5 --kick-z-frac 0.5
  --contact-damp 20,40 --track-gate 0.3 ...` with the `cam_policy` recipe); evaluate with
  `eval_cam_policy --nudges 2 --contact-damp 20,40 --nudge-up 0.5,1.5`; watch with
  `uv run mjpython -m sim.view_cam_policy --policy outputs/controller/cam_bounce/last.pt --nudge 2.5 --nudge-up 0.5,1.5`.

### Next: juggling (reward to try, iteration 2)

Goal: keep the ball bouncing on the paddle for as long as possible without dropping it. Not a target height: the
score is the time the ball stays juggled.

- **Episode**: starts with the ball dropped onto the paddle from ~10 cm; long episodes (20-30 s) so that time
  juggled can grow; no kicks needed at first (the bouncing is the task), random kicks later as a robustness test.
- **Juggling state**: a bounce counts when the ball leaves the blade and its flight peaks at least ~3 cm above its
  resting position (no credit for tiny vibrations). The ball is *juggled* while the time since the last counted bounce
  is under ~1 s.
- **Reward**: +1 per step while juggled, so the return is the time juggled (x 30 steps/s). Resting the ball counts
  as stopping: after ~1 s without a bounce the episode ends (no reward for holding it). -5 on a drop, as now.
  Small shaping kept from balancing: centring of the landing point (keeps the ball from drifting to the edge),
  the smoothness and home-pose penalties (reduced: hitting the ball needs quick moves).
- **Metrics**: time juggled before a drop or a stall, bounces per episode, bounce height spread.
- **Critic extra (training only)**: ball lift and vertical velocity (already there), time since the last bounce.
- **One policy for both?** To keep balancing in the same network, add a mode input (0 = balance, 1 = juggle) and mix
  both kinds of episode, as iteration 1 mixes sideways and upward kicks; start from the iteration-1 policy with zero
  weights for the new input.
- Watch for: hitting the ball needs fast paddle moves (action limit 0.04 rad per step, soft randomized servos) and a
  camera that follows a ball moving fast in the image (tracker distance fallback, `--track-gate`).

### Juggling (iteration 2): what happened

Implemented as planned (`sim/env.py` juggling episodes, `CamPolicyEnv(juggle=True)` mode input, `eval_cam_policy
--juggle`), fine-tuned from `cam_bounce`. Starting point: 0.9 s juggled on average, 0/90 to 30 s.

| run | change | result (training, 10 s episodes) |
|---|---|---|
| v1 | 70% juggling + 30% balance, smoothness 0.3, -5 on a drop | balance 56 -> 32%, juggling flat 1.0 -> 1.6 s: forgetting |
| v2 | + KL anchor to the bouncing policy on balance states, smoothness 0.05 when juggling, straight-hit bonus | balance held (43-48%), juggling 0.95 -> 1.34 s in 900 iterations; stalls 32 -> 55%: it learned to catch and hold |
| v3 | no -5 in juggling episodes (a drop and a stall both just end the score) | stalls still rose; juggling flat ~1.6 s |
| juggle-only | 100% juggling, no anchor, std 0.15 | 1.2 -> 2.9 s, 2.2 -> 7.2 bounces in 300 iterations |

- The arm can juggle physically: a scripted hit sends the ball 27-30 cm up. The hard part is precision (the scripted
  hitter always dropped it 10-23 cm to the side): soft, straight hits.
- **Two tasks in one network was the obstacle**, not juggling: the same start learned 3x faster alone. The anchor kept
  balancing but also pulled juggling toward "catch and settle"; a one-number mode input is a weak task signal.
- Reward pitfall: with -5 for a drop and nothing for a stall, holding the ball was the safe choice. The score is the time
  juggled, so a drop and a stall must cost the same.

### Multi-task: one network for several skills

Why one network struggles with two tasks, and what is known to work (multi-task RL literature):

1. **Conflicting gradients**: the tasks' updates push shared weights in opposite directions. PCGrad removes the
   conflicting component of each task's gradient before the update (Yu et al. 2020, arXiv:2001.06782).
2. **Different reward scales**: the task with larger returns dominates learning. PopArt normalises values and advantages
   per task (Hessel et al. 2019, arXiv:1809.04474). Cheapest version here: normalise advantages per task (not over the
   mixed batch) and give each task its own value head.
3. **Weak task conditioning**: one mode input among ~50. Stronger: per-task output heads, FiLM (the task scales/shifts
   every hidden layer), soft modularization (Yang et al. 2020, arXiv:2003.13661). Gains over a plain task ID are modest
   on Meta-World and vary between papers.
4. **Train separately, then combine** (the most reliable here): policy distillation trains one expert per task with
   RL, then a single network by supervised learning to copy each expert on its own task (Rusu et al. 2015,
   arXiv:1511.06295). Distral does it during RL: task policies kept close to a shared distilled one (Teh et al. 2017,
   arXiv:1707.04175). The KL anchor above was a one-sided version of this.

**Plan: distillation, DAgger style.**
1. Experts: balancing = `cam_bounce` (exists); juggling = the juggle-only policy (training).
2. Student: one actor with the mode input, same size as now (later: the MoE below).
3. Loop: run the *student* in both kinds of episode (simulated detections, all the randomization); label every state
   with the matching expert's action (mode 0 -> balancing expert, 1 -> juggling expert); train the student by regression
   on all states collected so far (MSE on the action mean, or KL with the experts' std). Running the student, not the
   experts, matters: it collects the states the student itself drifts into, so it learns to recover from its own
   mistakes (DAgger, Ross et al. 2011). Mix in some expert-driven rollouts at the start.
4. Check: each mode's evaluation vs its expert (balance: sideways and upward-kick tests; juggling: time juggled).
5. Optional PPO fine-tune of the student with per-task advantage normalisation, per-task value heads, PCGrad.
Supervised, fast, and no reward interference between the tasks.

**Idea: a mixture of experts (MoE) policy, to keep adding tasks.**
- Actor = K expert MLPs + a gate. The gate (small MLP on the task input, optionally the state) outputs weights over the
  experts; the action is the weighted sum of the experts' outputs (soft), or the top-1 expert (hard).
- Why: a new task gets a new expert; the old experts can be frozen, so old skills cannot be forgotten, and the gate
  learns when to use which. With a state-dependent soft gate, experts can be shared across tasks (a "catch the falling
  ball" expert used by both bouncing and juggling).
- Training: (a) distil the current experts into the MoE (each expert initialised from or distilled toward one task
  expert, the gate trained on the mode); (b) a new task: add an expert, freeze the rest, train the new expert and the
  gate (RL, then distil). Usual MoE care: a load-balancing term so one expert does not take every input, and a little
  noise in the gate early on.
- Related work: soft modularization (routing over modules, arXiv:2003.13661), mixture of orthogonal experts (MOORE,
  arXiv:2311.11385), parameter-compositional multi-task RL (PaCo, arXiv:2210.11653), progressive networks (frozen
  columns per task with lateral connections; Rusu et al. 2016).
- Cost: K x the actor (still tiny: K = 4 experts of 2 x 256 ~ 4 x 80k parameters, microseconds on the robot computer).
  Start with hard routing by task (= separate policies behind one interface), then soft, state-dependent gating once
  there are tasks that share skills.

### Goal: play ping pong (several tasks, one robot)

Balancing, bouncing and juggling are the first steps toward playing table tennis. A game is a sequence of distinct
skills, which is why the multi-task work (distillation, mixture of experts) matters: each skill can be an expert, and a
gate (or a high-level policy) picks the skill from the game state.

**The skills in a rally**, roughly in the order to build them:

| skill | what it needs | builds on |
|---|---|---|
| control the ball on the paddle | done: balance, catch upward kicks | — |
| juggle | done (in progress): repeated straight hits, time in the air | catching |
| aimed hit | send the ball to a target point (direction + height), not only straight up | juggling |
| receive / return | hit an incoming ball (thrown or launched) back over a net to a target region | aimed hit, tracking a fast ball |
| serve | toss the ball off the paddle, then hit it so it bounces on our side, over the net, then on theirs | juggling + aimed hit |
| rally | return repeatedly; recover to a ready pose between hits | receive, positioning |
| game | serve/receive rules, scoring, an opponent (a ball launcher, a scripted opponent, or self-play) | all of the above |

**What the simulation needs**, beyond what exists:
- A table and a net (and the ITTF bounce: the ball must bounce on the table), placed within the arm's reach — likely a
  scaled-down table for the SO-101's ~40 cm reach.
- An incoming ball: a launcher with random speed, direction and spin, for receiving; later an opponent.
- Ball flight physics: air drag and the Magnus force from spin (MuJoCo has no aerodynamics for a sphere by default;
  add them as forces each step), and spin transfer at paddle and table contacts (rubber friction).
- Perception for a fast ball over the whole table: the post camera may not cover it; a second camera, or one higher and
  further back; 30 Hz may be too slow for a returned ball (~3-6 m/s) - check exposure and motion blur on the real camera.
- Timing: a hit has to happen at the right moment, so latency matters more than for balancing (see "Real arm: latency").

**Hardware limits to check early**: the SO-101's joint speeds and torques (can it swing a paddle fast enough for a
return?), reach, and the paddle mount. A slower game (lighter returns, a mini table) may be the realistic target.

**How to train it**: one expert per skill (RL, each with its own reward), distilled DAgger-style into one policy with a
skill input, or a mixture of experts with a gate driven by the game phase; possibly a high-level policy choosing the
skill (hierarchical RL). Each new skill = a new expert, old ones frozen. Evaluate every skill on its own test, and the
whole game on rallies completed.

## 2. PPO → GRPO

Keep the ball task, so PPO's 97.5% is the target to compare against.

**What changes:** GRPO (Group Relative Policy Optimization, from LLM training) drops PPO's value network (the critic).
It runs a *group* of G attempts from the same start and scores each attempt against the group:
`A_i = (R_i − mean(R)) / std(R)`. The policy update stays PPO's clipped objective, plus an optional KL penalty
toward a reference policy.

Work:
- New `sim/grpo.py` next to `sim/ppo.py`: policy-only network (no value head), group advantages, clipped loss + KL term.
- Grouped rollouts in `sim/train_controller.py`: reset G environments with the **same seed**, so they share the start
  state and domain randomization, and run them to the end of the episode. The current loop collects fixed 128-step
  chunks across episode boundaries, so this is the main change.
- Advantage per episode, applied to every step of it. Optionally try per-step variants (reward-to-go normalized
  across the group at each time step) to reduce noise.
- Log the same metrics as PPO and compare learning curves (success rate vs environment steps).

Expectation: with 300-step episodes and one score per episode, the training signal is much noisier than with PPO's
per-step estimates, so expect lower sample efficiency. Finding out how much is the experiment.

**Done when:** GRPO trains a controller you can compare with PPO on `sim.eval_policy`, given the same step budget.

## 3. New task: tray + glass of water

A new task, done last, once the training tools are trusted.

### Model (MJCF)
- Replace the paddle with a tray (thin box + low rim) held in the gripper.
- Glass: a free-moving cylinder (or mesh) with realistic mass, friction and a high centre of mass.
- **Payload:** SO-101 servos are weak. Tray + glass + water must stay within what the real arm can hold. Use real masses.

### The water
MuJoCo 4.0.0 has no liquid solver. Its docs say *"Proper simulation of fluid dynamics is beyond the scope of MuJoCo"*.
Its fluid models are drag forces on solid bodies, and 4.0.0's new deformables are soft *solids* (elasto-plastic flex).
Options, simplest first:

| Option | How | Trade-off |
|---|---|---|
| Rigid glass | glass + water = one rigid body | fast; no slosh |
| Slosh model | rigid glass + a hidden mass on a spring/pendulum joint inside it (standard engineering model of liquid slosh) | fast, captures the main effect; tune frequency and damping |
| Many small spheres | dozens of small spheres in the glass (grain-like stand-in for liquid) | looks like spilling; much slower, harder to train |
| Another simulator | a physics engine with real liquid simulation (e.g. particle-based fluids) | real water; leaves MuJoCo, big port |

Recommended: start rigid, then add the slosh model.

### Environment
New `TrayEnv` based on `sim/env.py`:
- `reset`: glass at a random spot on the tray (and random fill level / mass if you add slosh).
- Observation: glass position on the tray, tilt, sliding speed (+ slosh state if available), joints, recent actions.
- Reward: glass upright, not sliding, away from the edge, smooth actions.
- Termination: glass tipped past a threshold angle, off the tray, or spilled.
- Keep the domain randomization that made sim-to-real work (latency, soft servos, observation noise).

### Make the task harder than "hold still"
A glass standing on a level tray is stable on its own, so a policy that does nothing would succeed. Give the arm a job:
- **Waiter:** carry the tray between random target poses without spilling or tipping.
- **Disturbance rejection:** keep the glass steady while the arm base or the glass gets random kicks.

### Training
1. Train with PPO first (known to work), then GRPO from step 2.
2. Perception (seeing the glass) last, if at all. The orange-blob detector won't find a glass. Training and evaluating
   on the true sim state is fine for learning RL.

## Practical notes

- Full training took ~2 h on a Linux machine with a GPU (README). On a Mac it will be slower. For every change,
  start with a short run (`--iters 200`) and check that the reward goes up before committing to a full run.
- On macOS the live viewer needs `mjpython`: `uv run mjpython -m sim.view_controller --rand --seconds 20`.
- All experiments in one TensorBoard (training curves, estimator training, evaluations vs latency, stress tests,
  results notes), rebuilt from the logs in `outputs/`:
  `uv run --extra tensorboard python -m sim.to_tensorboard`, then
  `uv run --extra tensorboard tensorboard --logdir outputs/tensorboard`.

## Real arm: latency (to do when the arm arrives)

In sim, **latency** = the whole control loop's delay in 33 ms steps (30 Hz): the action chosen now is applied N steps
later (`BalanceEnv._action_delay`; the arm + camera policy delays its pan/tilt commands the same way). On the robot it
is the sum of camera exposure + USB transfer, detection + policy, the servo bus, and the servo starting to move.
How softly a servo then follows the command is separate (`servo_rand`: stiffness, damping, force limit).
Training draws 0-4 steps per episode, so the policies cope with any fixed delay up to 133 ms without being told which;
evaluations fix it per row. Latency 4 with pushes is the weak case for every setup (it acts on 133 ms old images).

**Measure it** (`hardware/run_real.py` does not yet):
1. Step test: command a small sudden joint move, timestamp the command and the first camera frame where the paddle
   moves; ~20 repeats, mean and spread. That is the full loop delay.
2. Per-stage timestamps every frame (capture, detection, policy, command sent, encoder read-back): where the time goes,
   and whether it is steady or jumps.
Divide by 33 ms. Above 4 steps the policies are outside their training range: retrain with a larger `--max-latency`.

**Adapting to it**, simplest first:
- a. Make it constant: pad every loop to a fixed total (e.g. always 2 steps). A known, steady delay is much easier than
  a variable one. No retraining.
- b. Tell the policy: the critic already gets the latency in training; give it to the actor too, and on the robot feed
  the value measured live from the timestamps. One input, ~35 min retraining.
- c. Let the policy infer it: longer history of its commands and the joint encoders, a recurrent network or a small
  learned estimator of the delay (rapid motor adaptation, as on legged robots). Pays off if the delay changes during a
  run, or for bouncing.
Plan: measure; if steady, (a) padded to the measured value and compare with the evaluation row for that latency; if it
jitters frame to frame (USB cameras can), (b) with live timestamps. (b) can be tried in sim before the arm arrives.
