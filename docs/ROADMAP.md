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
