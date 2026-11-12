"""SO-101 ball-on-paddle balancing in MuJoCo (Gymnasium API).

obs_mode:
    "controller"  flat float32 vector for the PPO controller:
                  [0:4] ball (u, w) / blade radius, (vu, vw) / 0.5 m/s, in the blade frame
                  [4:9] joint angles - home, / 0.2 rad    [9:14] joint velocities * 0.1    [14:] last act_hist actions, newest first
    "image"       {"image": wrist camera (96x128 RGB)}
Action: 5 joint-angle deltas in [-1, 1] (x action_scale rad), 30 Hz. The gripper stays closed on the paddle.

Kicks (off by default, kick_max > 0): mid-episode pushes that add a sideways velocity to the ball, one at a random step
in each kick_every-second window between kick_grace s after the start and kick_margin s before the end, speed uniform in
[kick_min, kick_max] m/s, random direction. The schedule (self.kicks) is fixed at reset and comes from the reset seed
alone, so episodes reset with the same seed get the same kicks whatever the policy does. The policy is not told.

Bouncing (off by default): kick_z_max > 0 gives the kicks of a kick_z_frac share of the episodes an upward component,
uniform in [kick_z_min, kick_z_max] m/s, so the ball flies off the paddle and bounces when it lands; contact_damp makes
the ball/blade contact elastic (restitution ~0.75-0.86 instead of a dead ~0.1). With either on, the ball may be in the
air: the episode ends if it falls past the blade, lands off it, or flies far away; success also needs it back on the
blade at the end. The reward is unchanged (centred, slow ball): catch it softly and settle it.

Juggling (per episode: reset(options={"juggle": True, "drop": h})): the ball is dropped from h above its resting
position, no kicks, and the task is to keep it bouncing. A bounce counts when its flight peaks >= JUGGLE_MIN_APEX above
the resting position; the ball is juggled while the last counted bounce (or the drop) is under JUGGLE_STALL_S old.
Reward per step: 1 while juggled + JUGGLE_CENTRE_W * centring - the home penalty, + a straight-hit bonus at each counted
bounce (largest when it peaks over the centre) and at each impact (blade level, moving straight up). Smoothness: the
full smooth_penalty and a joint-acceleration penalty everywhere except the hit window (ball within HIT_WINDOW_LIFT of
the blade), where only juggle_smooth applies: one clean move at the hit, no shaking in between. The episode ends on
a drop or a stall (no counted bounce for JUGGLE_STALL_S: holding the ball earns nothing), with no extra penalty for
either: the score is the time juggled. Success: juggled to the end.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # headless GPU rendering

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

SCENE = Path(__file__).parent.parent / "assets" / "so101" / "scene_balance.xml"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# blade level, centre at ~(0.415, 0, 0.297) m, far from all joint limits
HOME_Q = np.array([0.0, -0.65, 0.15, 0.5, -1.52212])


# Estimator position error (cm, measured in sim; / 8 cm = controller units): per-episode bias, slow drift (~60 steps), per-frame jitter
BN_BIAS = np.array([0.3, 1.2]) / 8
BN_OU = np.array([0.15, 0.5]) / 8
BN_WHITE = np.array([0.03, 0.06]) / 8
BN_A = float(np.exp(-1 / 60))
FD_ALPHA = 0.5
CONTROL_HZ = 30
CAM_TRACK_RATE = 0.2  # rad per control step (6 rad/s): pan/tilt servo speed limit of the tracking camera
CAM_AIM_LIMIT = np.array([0.6, 0.5])  # rad, pan/tilt range around the nominal aim (servo limits): covers the paddle's
# whole workspace (blade centre ~0.33 rad below nominal), never reaches the post below the camera
CAM_FEAT_SCALE = np.array([0.004, 0.004, 0.004, np.deg2rad(2.5), np.deg2rad(2.5), np.deg2rad(2.5), 4.0])  # pose offset -> ~unit
# Elastic ball/blade contact, MuJoCo's direct form solref = (-stiffness, -damping). Drop tests on a still paddle:
# damping 20 -> restitution e ~0.86, 30 -> ~0.80, 40 -> ~0.75 (the default solref "0.004 1": ~0.1, the ball lands dead).
BOUNCE_STIFF = 40000.0
JUGGLE_MIN_APEX = 0.03  # m above the resting position: a smaller hop is not a bounce
JUGGLE_STALL_S = 1.0  # s without a counted bounce: stalled
JUGGLE_CENTRE_W = 0.2  # centring shaping in juggling episodes
JUGGLE_STRAIGHT_W, JUGGLE_STRAIGHT_SIGMA = 1.0, 0.03  # bonus per counted bounce: W * exp(-(offset at the apex / SIGMA)^2)
# Height command (juggle_target=(lo, hi)): a target apex height, redrawn every 1-3 counted bounces; bonus per bounce
RELEASE_W, RELEASE_SPAN = 4.0, 1.2  # at each release (ball leaves the blade upward): W * (1 - min(|vz - sqrt(2 g target)| / SPAN, 1));
# linear, not a narrow bell: a bell around the right speed gave no signal to a policy 0.8 m/s too slow
RELEASE_MIN_VZ = 0.3  # m/s: slower separations are contact chatter, not a hit
HEIGHT_W, HEIGHT_SPAN = 2.0, 0.15  # W * (1 - min(|apex - target| / SPAN, 1)), m (linear for the same reason)
# Clean hits: smooth everywhere except a short window around each contact, and a reward for a level, vertical hit.
HIT_WINDOW_LIFT = 0.035  # m: the ball this close to the blade = the hit window (juggle_smooth applies; else smooth_penalty)
JUGGLE_ACC_W = 0.05  # outside the window: reward -= W * sum((joint velocity change per step, rad/s)^2): no shaking
HIT_LEVEL_W, HIT_LEVEL_SIGMA = 0.5, np.deg2rad(5.0)  # at each ball/blade impact: blade normal close to vertical
HIT_LATERAL_W, HIT_LATERAL_SIGMA = 0.5, 0.1  # ... and blade moving straight up (sideways speed, m/s)
KICK_SEED_SALT = 0x6B69636B  # "kick": the kick generator is seeded from (seed, salt), independent of the env's main rng


def kick_schedule(rng: np.random.Generator, n_steps: int, every_s: float, grace_s: float, margin_s: float,
                  speed: tuple[float, float], up: tuple[float, float] | None = None) -> dict[int, np.ndarray]:
    """{step: (vx, vy) m/s}: one kick at a random step in each every_s window inside [grace_s, end - margin_s).
    up = (lo, hi): each kick also gets an upward speed in that range, {step: (vx, vy, vz)}."""
    lo_speed, hi_speed = speed
    if every_s <= 0 or hi_speed <= 0:
        return {}
    start, stop = round(grace_s * CONTROL_HZ), n_steps - round(margin_s * CONTROL_HZ)
    window = max(1, round(every_s * CONTROL_HZ))
    kicks = {}
    for lo in range(start, stop, window):
        t = int(rng.integers(lo, min(lo + window, stop)))
        ang, v = rng.uniform(0, 2 * np.pi), rng.uniform(lo_speed, hi_speed)
        kicks[t] = v * np.array([np.cos(ang), np.sin(ang)])
        if up is not None:
            kicks[t] = np.r_[kicks[t], rng.uniform(*up)]
    return kicks


class FdVel:
    """Ball velocity from successive positions: smoothed finite differences. The same code runs in training and on the robot."""

    def __init__(self, blade_radius: float = 0.08, dt: float = 1 / 30):
        self.k = blade_radius / dt / 0.5
        self.reset()

    def reset(self):
        self.p, self.v = None, np.zeros(2)

    def update(self, p) -> np.ndarray:
        p = np.asarray(p, np.float64)
        if self.p is not None:
            self.v = (1 - FD_ALPHA) * self.v + FD_ALPHA * (p - self.p) * self.k
        self.p = p.copy()
        return np.r_[p, self.v]


class BalanceEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        obs_mode: str = "controller",
        image_hw: tuple[int, int] = (96, 128),
        camera: str = "wrist",  # model camera rendered in image mode: "wrist" (on the gripper) or "ego" (post on the base)
        max_steps: int = 300,  # 10 s
        action_scale: float = 0.04,  # rad per step at |a| = 1
        domain_rand: bool = False,
        ball_start_radius: float = 0.02,  # m, ball starts within this distance of the blade centre
        ball_start_speed: float = 0.03,  # m/s, ball starts with a random velocity up to this per axis
        light_model: bool | None = None,  # strip visual-only meshes (default: in controller mode)
        success_radius: float = 0.06,  # m, success = ball within this of the centre at the end
        grip_rot_deg: float = 0.0,  # paddle rotation about the tool axis (real grip: -90, blade edge-on to the camera)
        max_latency: int = 2,  # randomized control latency: 0..max_latency extra steps
        home_penalty: float = 0.0,  # reward -= home_penalty * sum(((q - home) / 20 deg)^2)
        smooth_penalty: float = 0.01,  # reward -= smooth_penalty * sum((a - previous a)^2): discourages jittery actions
        servo_rand: bool = False,  # randomize servo stiffness, damping and force limit
        cam_rand: float = 1.0,  # scale of the camera pose/FOV randomization
        cam_track: bool = False,  # pan/tilt camera: track_detection() steers it toward the detected ball (speed-limited)
        ball_noise: float = 0.0,  # >0: controller sees estimator-like noisy positions + FdVel velocity (1 = measured error)
        act_hist: int = 1,  # past actions in the controller obs (covers commands still in flight under latency)
        kick_min: float = 0.0,  # m/s, kick speed range (see module docstring); kick_max = 0: no kicks
        kick_max: float = 0.0,
        kick_every: float = 4.0,  # s, one kick per window of this length
        kick_grace: float = 1.0,  # s, no kick before this
        kick_margin: float = 1.0,  # s, no kick in the last this-many seconds (time to recover before the success check)
        kick_z_min: float = 0.0,  # m/s, upward kick component range (bouncing, see module docstring); kick_z_max = 0: off
        kick_z_max: float = 0.0,
        kick_z_frac: float = 1.0,  # share of the episodes whose kicks get the upward component
        contact_damp: tuple[float, float] | None = None,  # elastic ball/blade contact, damping drawn per episode
        juggle_smooth: float | None = None,  # smoothness penalty in the hit window of juggling episodes (None: smooth_penalty)
        juggle_calm: float | None = None,  # smoothness penalty between hits in juggling episodes (None: smooth_penalty)
        juggle_acc: float = JUGGLE_ACC_W,  # joint-acceleration penalty between hits in juggling episodes
        juggle_target: tuple[float, float] | None = None,  # juggling: commanded apex height range (m), redrawn every 1-3 bounces
        height_w: float = HEIGHT_W,  # weight of the bonus for a bounce at the commanded height
        seed: int | None = None,
    ):
        if not 0.0 <= kick_min <= kick_max:
            raise ValueError(f"need 0 <= kick_min <= kick_max, got {kick_min}, {kick_max}")
        assert obs_mode in ("image", "controller")
        self.obs_mode, self.image_hw = obs_mode, image_hw
        self.max_steps, self.action_scale, self.domain_rand = max_steps, action_scale, domain_rand
        self.ball_start_radius, self.ball_start_speed, self.success_radius = ball_start_radius, ball_start_speed, success_radius
        self.max_latency, self.act_hist = max_latency, act_hist
        self.home_penalty, self.servo_rand, self.ball_noise, self.cam_rand = home_penalty, servo_rand, ball_noise, cam_rand
        self.smooth_penalty, self.cam_track = smooth_penalty, cam_track
        self._fd = FdVel()
        self._ball_obs = np.zeros(4)
        self._hist = np.zeros((act_hist, 5))
        self.rng = np.random.default_rng(seed)
        self.kick_speed, self.kick_every, self.kick_grace, self.kick_margin = (kick_min, kick_max), kick_every, kick_grace, kick_margin
        self.kicks: dict[int, np.ndarray] = {}
        self.kick_z, self.kick_z_frac, self.contact_damp = (kick_z_min, kick_z_max), kick_z_frac, contact_damp
        self.airborne_ok = kick_z_max > 0 or contact_damp is not None  # the ball may leave the blade
        self.kick_z_on = False  # this episode's kicks have an upward component
        self.juggle = False  # this episode is a juggling episode (reset option)
        self.juggle_smooth, self.juggle_calm, self.juggle_acc = juggle_smooth, juggle_calm, juggle_acc
        self.juggle_target = tuple(juggle_target) if juggle_target is not None else None
        self.target_apex = 0.0  # the commanded apex height of the next bounce (0 = none)
        self.height_w = height_w

        self.grip_rot_deg = grip_rot_deg
        # wrist roll compensates the grip rotation, so the blade has the same world pose
        self.home_q = HOME_Q.copy(); self.home_q[4] -= np.deg2rad(grip_rot_deg)
        self.m = self._load_model(light=(obs_mode == "controller") if light_model is None else light_model, grip_rot_deg=grip_rot_deg)
        self.m.opt.timestep = 1.0 / 480.0
        self.n_substeps = 16
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.q_adr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.v_adr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.act_ids = np.array([m.actuator(j).id for j in ARM_JOINTS])
        self.grip_act = m.actuator("gripper").id
        self.ball_qadr = m.joint("ball_joint").qposadr[0]
        self.ball_vadr = m.joint("ball_joint").dofadr[0]
        self.ball_body, self.ball_geom = m.body("ball").id, m.geom("ball_geom").id
        self.blade_geom, self.grip_body = m.geom("paddle_blade").id, m.body("gripper").id
        self.floor_geom = m.geom("floor").id
        self.cam_id = m.camera(camera).id
        self.blade_radius = float(m.geom_size[self.blade_geom][0])
        self.ball_radius = float(m.geom_size[self.ball_geom][0])
        jids = [m.joint(j).id for j in ARM_JOINTS]
        self.q_lo, self.q_hi = m.jnt_range[jids, 0].copy(), m.jnt_range[jids, 1].copy()
        self._nominal = dict(
            cam_pos=m.cam_pos[self.cam_id].copy(), cam_quat=m.cam_quat[self.cam_id].copy(), cam_fovy=float(m.cam_fovy[self.cam_id]),
            ball_rgba=m.geom_rgba[self.ball_geom].copy(), ball_mass=float(m.body_mass[self.ball_body]),
            ball_inertia=m.body_inertia[self.ball_body].copy(), ball_fric=m.geom_friction[self.ball_geom].copy(),
            blade_fric=m.geom_friction[self.blade_geom].copy(), headlight=m.vis.headlight.diffuse.copy(),
            ambient=m.vis.headlight.ambient.copy(),
            gain=m.actuator_gainprm[self.act_ids, 0].copy(), bias=m.actuator_biasprm[self.act_ids, :3].copy(), frc=m.actuator_forcerange[self.act_ids].copy(),
        )
        self._renderer = None
        self._cam_dq = np.array([1.0, 0, 0, 0])  # mounting error rotation (randomized)
        self._cam_feat = np.zeros(7)
        R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, self._nominal["cam_quat"])
        self._aim0 = self._yaw_pitch(-R0.reshape(3, 3)[:, 2])  # nominal looking direction, in the camera's parent body
        self._aim = self._aim0.copy()
        self._aim_want = None

        h, w = image_hw
        if obs_mode == "controller":
            self.observation_space = spaces.Box(-np.inf, np.inf, (14 + 5 * act_hist,), np.float32)
        else:
            self.observation_space = spaces.Dict({"image": spaces.Box(0, 255, (h, w, 3), np.uint8)})
        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)

        self._q_target = self.home_q.copy()
        self._prev_action = np.zeros(5)
        self._delay_buf: list[np.ndarray] = []
        self._action_delay = 0
        self._img_aug = (1.0, 0.0, 0.0)
        self.t = 0

    @staticmethod
    def _load_model(light: bool, grip_rot_deg: float = 0.0):
        """light=True drops visual-only geoms, meshes, materials and textures (~15 MB less per env; collision geometry is untouched)."""
        if not light and grip_rot_deg == 0.0:
            return mujoco.MjModel.from_xml_path(str(SCENE))
        spec = mujoco.MjSpec.from_file(str(SCENE))
        if grip_rot_deg != 0.0:  # rotate the paddle about the gripper z (tool) axis
            h = np.deg2rad(grip_rot_deg) / 2
            spec.body("paddle").quat = [np.cos(h), 0.0, 0.0, np.sin(h)]
        if not light:
            return spec.compile()
        for g in [g for g in spec.geoms if g.contype == 0 and g.conaffinity == 0 and g.group != 3]:
            spec.delete(g)
        used = {g.meshname for g in spec.geoms if g.meshname}
        for mesh in [x for x in spec.meshes if x.name not in used]:
            spec.delete(mesh)
        for g in spec.geoms:
            g.material = ""
        for mat in list(spec.materials):
            spec.delete(mat)
        for tex in [t for t in spec.textures if t.name != "skybox"]:
            spec.delete(tex)
        return spec.compile()

    def blade_frame(self):
        """Blade centre, normal, and in-plane axes u (paddle width) and w (along the tool axis)."""
        d = self.d
        c = d.geom_xpos[self.blade_geom].copy()
        n = d.geom_xmat[self.blade_geom].reshape(3, 3)[:, 2].copy()
        g = d.xmat[self.grip_body].reshape(3, 3)
        th = np.deg2rad(self.grip_rot_deg)  # paddle width axis = gripper x rotated by the grip angle
        return c, n, g @ np.array([np.cos(th), np.sin(th), 0.0]), -g[:, 2].copy()

    def ball_in_blade(self):
        """Ball offset in the blade plane (u, w), height above it h, and speed, in m and m/s."""
        c, n, u, w = self.blade_frame()
        rel = self.d.xpos[self.ball_body] - c
        vel = self.d.qvel[self.ball_vadr : self.ball_vadr + 3]
        return float(rel @ u), float(rel @ w), float(rel @ n), float(np.linalg.norm(vel))

    def _randomize(self):
        m, n, r = self.m, self._nominal, self.rng
        # camera mount pose and FOV
        k = self.cam_rand
        m.cam_pos[self.cam_id] = n["cam_pos"] + r.uniform(-0.004, 0.004, 3) * k
        dq = np.zeros(4)
        mujoco.mju_axisAngle2Quat(dq, r.normal(size=3) / 1.0, np.deg2rad(r.uniform(0, 2.5) * k))
        mujoco.mju_mulQuat(m.cam_quat[self.cam_id], n["cam_quat"], dq)
        m.cam_fovy[self.cam_id] = n["cam_fovy"] + r.uniform(-4, 4) * k
        self._cam_dq = dq
        rv = np.zeros(3); mujoco.mju_quat2Vel(rv, dq, 1.0)
        self._cam_feat = np.r_[m.cam_pos[self.cam_id] - n["cam_pos"], rv, m.cam_fovy[self.cam_id] - n["cam_fovy"]] / CAM_FEAT_SCALE
        # ball colour, mass, friction; blade friction
        rgba = n["ball_rgba"].copy(); rgba[:3] = np.clip(rgba[:3] * r.uniform(0.75, 1.1, 3), 0, 1); m.geom_rgba[self.ball_geom] = rgba
        s = r.uniform(0.85, 1.2)
        m.body_mass[self.ball_body] = n["ball_mass"] * s; m.body_inertia[self.ball_body] = n["ball_inertia"] * s
        m.geom_friction[self.ball_geom] = n["ball_fric"] * np.array([r.uniform(0.6, 1.4), 1, 1])
        m.geom_friction[self.blade_geom] = n["blade_fric"] * np.array([r.uniform(0.6, 1.4), 1, 1])
        m.vis.headlight.diffuse[:] = n["headlight"] * r.uniform(0.5, 1.3)
        m.vis.headlight.ambient[:] = n["ambient"] * r.uniform(0.5, 1.5)
        # control latency, servos, image contrast/brightness/noise
        self._action_delay = int(r.integers(0, self.max_latency + 1))
        if self.servo_rand:  # per joint: stiffness x0.03-1 (log-uniform), damping x0.3-1.5, force limit x0.6-1
            kp = n["gain"] * np.exp(r.uniform(np.log(0.03), 0.0, 5)); kv = -n["bias"][:, 2] * r.uniform(0.3, 1.5, 5)
            m.actuator_gainprm[self.act_ids, 0] = kp; m.actuator_biasprm[self.act_ids, 1] = -kp; m.actuator_biasprm[self.act_ids, 2] = -kv
            m.actuator_forcerange[self.act_ids] = n["frc"] * r.uniform(0.6, 1.0, (5, 1))
        self._img_aug = (r.uniform(0.8, 1.2), r.uniform(-15, 15), r.uniform(0, 4))
        # paddle body colour: black to bare wood
        mid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "paddle_black_mat")
        if mid >= 0:
            wood = np.array([0.62, 0.40, 0.22]) * r.uniform(0.7, 1.15, 3)
            dark = r.uniform(0.02, 0.12, 3)
            m.mat_rgba[mid, :3] = np.clip(dark + r.uniform(0, 1) * (wood - dark), 0, 1)
        self._randomize_background()

    def _randomize_background(self):
        """Random floor (plain / checker / wood-like) and sky gradient, so the camera cannot rely on the sim background."""
        m, r = self.m, self.rng
        if m.ntex < 2:
            return
        import colorsys, cv2

        def colour():  # any hue, mostly muted
            return np.array(colorsys.hsv_to_rgb(r.uniform(), r.uniform(0, 0.7), r.uniform(0.2, 0.95)))

        def noise(h, w, cells):
            g = r.uniform(-1, 1, (cells, cells)).astype(np.float32)
            return cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)[..., None]

        gid = 1  # floor texture
        h, w = int(m.tex_height[gid]), int(m.tex_width[gid])
        c1, c2 = colour(), colour()
        if r.uniform() < 0.5:  # low-contrast surface
            c2 = np.clip(c1 + r.uniform(-0.12, 0.12, 3), 0, 1)
        style = r.choice(["plain", "checker", "wood"], p=[0.35, 0.35, 0.30])
        yy, xx = np.mgrid[0:h, 0:w]
        if style == "plain":
            mix = np.zeros((h, w, 1), np.float32)
        elif style == "checker":
            n = int(r.integers(2, 17))
            mix = (((yy * n // h) + (xx * n // w)) % 2).astype(np.float32)[..., None]
        else:
            f, ph = r.uniform(3, 25), r.uniform(0, 6.28)
            mix = (0.5 + 0.5 * np.sin(2 * np.pi * f * (xx if r.uniform() < 0.5 else yy) / w + ph + 2 * noise(h, w, 6)[..., 0]))[..., None].astype(np.float32)
        img = c1 * (1 - mix) + c2 * mix
        img = img * (1 + r.uniform(0, 0.1) * noise(h, w, int(r.integers(4, 24))))
        adr = int(m.tex_adr[gid])
        m.tex_data[adr:adr + h * w * 3] = (np.clip(img, 0, 1) * 255).astype(np.uint8).reshape(-1)
        mid = int(m.geom_matid[self.floor_geom])
        m.mat_reflectance[mid] = r.uniform(0, 0.25)
        m.mat_texrepeat[mid] = r.uniform(1, 8, 2)
        # sky gradient
        sh, sw = int(m.tex_height[0]), int(m.tex_width[0])
        top, bot = colour(), colour()
        t = np.tile(np.linspace(0, 1, sw, dtype=np.float32)[:, None], (sh // sw, 1))[..., None]
        sky = top * (1 - t) + bot * t
        m.tex_data[0:sh * sw * 3] = (np.clip(np.broadcast_to(sky, (sh, sw, 3)), 0, 1) * 255).astype(np.uint8).reshape(-1)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(m, *self.image_hw)
        for tid in (0, gid):
            mujoco.mjr_uploadTexture(m, self._renderer._mjr_context, tid)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if self.domain_rand:
            self._randomize()
        m, d, r = self.m, self.d, self.rng
        if self.contact_damp is not None:
            m.geom_solref[self.ball_geom] = m.geom_solref[self.blade_geom] = (-BOUNCE_STIFF, -r.uniform(*self.contact_damp))
        opts = options or {}
        self.juggle = bool(opts.get("juggle", False))
        drop = float(opts.get("drop", 0.0)) if self.juggle else 0.0
        mujoco.mj_resetData(m, d)
        q0 = self.home_q + r.uniform(-0.03, 0.03, 5)
        d.qpos[self.q_adr] = q0
        self._q_target = q0.copy()
        d.ctrl[self.act_ids] = q0
        d.ctrl[self.grip_act] = 0.0
        mujoco.mj_forward(m, d)
        # ball near the blade centre, 3 mm above the surface
        c, n, u, w = self.blade_frame()
        ang, rad = r.uniform(0, 2 * np.pi), self.ball_start_radius * np.sqrt(r.uniform())
        pos = c + n * (m.geom_size[self.blade_geom][1] + self.ball_radius + 0.003 + drop) + u * rad * np.cos(ang) + w * rad * np.sin(ang)
        d.qpos[self.ball_qadr : self.ball_qadr + 3] = pos
        d.qpos[self.ball_qadr + 3 : self.ball_qadr + 7] = [1, 0, 0, 0]
        d.qvel[self.ball_vadr : self.ball_vadr + 3] = r.uniform(-self.ball_start_speed, self.ball_start_speed, 3)
        mujoco.mj_forward(m, d)
        self.t, self._prev_action = 0, np.zeros(5)
        self._hist = np.zeros((self.act_hist, 5))
        self._delay_buf = [np.zeros(5)] * self._action_delay
        self.kicks = self._draw_kicks(seed)
        if self.juggle:  # the bouncing is the task: no kicks; the drop counts as the first bounce
            self.kicks, self.kick_z_on = {}, False
            self.n_bounces, self._last_bounce, self._vz_prev, self._apexes, self._apex_offsets = 0, 0, 0.0, [], []
            self._target_errs, self._release_errs = [], []
            self._new_target()
            self._impacts, self._jit_between, self._touching = [], [], False  # dropped from above: not touching yet
            self._qd_prev = self.d.qvel[self.v_adr].copy()
        if self.cam_track:  # start aimed at the blade centre (known from the arm's kinematics)
            self._aim, self._aim_want = self._yaw_pitch(self._parent_dir(self.d.geom_xpos[self.blade_geom])), None
            self._apply_aim()
        if self.ball_noise:
            k = r.uniform(0.5, 1.5) * self.ball_noise
            self._bn = dict(k=k, bias=r.normal(0, 1, 2) * BN_BIAS * k, ou=r.normal(0, 1, 2) * BN_OU * k, gain=1 + r.uniform(-0.1, 0.1, 2) * k)
            self._fd.reset()
            self._noisy_ball()
        return self._obs(), self._info()

    def _draw_kicks(self, seed: int | None) -> dict[int, np.ndarray]:
        if self.kick_speed[1] <= 0:
            return {}  # kicks off: draw nothing, so the main rng sequence is unchanged
        kseed = [seed, KICK_SEED_SALT] if seed is not None else [int(self.rng.integers(2**32)), KICK_SEED_SALT]
        krng = np.random.default_rng(kseed)
        self.kick_z_on = self.kick_z[1] > 0 and krng.uniform() < self.kick_z_frac  # no draw when off: old schedules unchanged
        return kick_schedule(krng, self.max_steps, self.kick_every, self.kick_grace, self.kick_margin, self.kick_speed,
                             up=self.kick_z if self.kick_z_on else None)

    # --- camera: mounting-error features (experiment: known camera pose) and ball tracking (pan/tilt) ---
    @staticmethod
    def _yaw_pitch(f):
        return np.array([np.arctan2(f[1], f[0]), np.arctan2(f[2], np.hypot(f[0], f[1]))])

    def _parent_dir(self, target):
        """Unit direction from the camera to a world point, in the camera's parent-body frame."""
        b = self.m.cam_bodyid[self.cam_id]
        R = self.d.xmat[b].reshape(3, 3)
        cam_world = self.d.xpos[b] + R @ self.m.cam_pos[self.cam_id]
        f = R.T @ (target - cam_world)
        return f / np.linalg.norm(f)

    def _apply_aim(self):
        """Point the camera along self._aim (yaw, pitch in the parent frame), image upright, then the mounting error."""
        y, p = self._aim
        f = np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
        x = np.cross(f, [0.0, 0.0, 1.0]); x /= np.linalg.norm(x)
        z = -f
        R = np.stack([x, np.cross(z, x), z], axis=1)
        q = np.zeros(4); mujoco.mju_mat2Quat(q, R.ravel())
        mujoco.mju_mulQuat(self.m.cam_quat[self.cam_id], q, self._cam_dq)
        mujoco.mj_camlight(self.m, self.d)  # the rendered camera pose (d.cam_xmat) follows now, not at the next mj_step

    def cam_features(self) -> np.ndarray:
        """The camera's mounting offset this episode (position, rotation vector, FOV), scaled to ~unit. Real arm: calibration."""
        return self._cam_feat.astype(np.float32)

    def track_detection(self, det) -> None:
        """Visual servo for the tracking camera: the detected ball's image offset -> a pan/tilt target, reached (speed-limited)
        during the next step. Uses the nominal FOV only, like a real tracker; no detection = hold the aim."""
        if not self.cam_track or det[3] == 0:
            return
        h, w = self.image_hw
        tv = np.tan(np.deg2rad(self._nominal["cam_fovy"]) / 2)
        self._aim_want = self._aim - np.array([np.arctan(det[0] * tv * w / h), np.arctan(det[1] * tv)])

    def cam_aim(self) -> np.ndarray:
        """Tracking camera pan/tilt relative to the nominal aim, rad / 0.5. Real arm: the pan/tilt servo encoders."""
        return ((self._aim - self._aim0) / 0.5).astype(np.float32)

    def step(self, action):
        if self.t in self.kicks:
            k = self.kicks[self.t]
            self.d.qvel[self.ball_vadr:self.ball_vadr + len(k)] += k
        a = np.clip(np.asarray(action, np.float64), -1, 1)
        self._delay_buf.append(a)
        a_applied = self._delay_buf.pop(0)
        self._q_target = np.clip(self._q_target + a_applied * self.action_scale, self.q_lo, self.q_hi)
        self.d.ctrl[self.act_ids] = self._q_target
        impact = release = None
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.m, self.d)
            if self.juggle:
                ev = self._contact_event()
                if ev is not None and ev[0] == "impact" and impact is None:
                    impact = ev[1]
                elif ev is not None and ev[0] == "release" and release is None:
                    release = ev[1]
        self.t += 1
        if self.cam_track and self._aim_want is not None:
            self._aim = self._aim + np.clip(self._aim_want - self._aim, -CAM_TRACK_RATE, CAM_TRACK_RATE)
            self._aim = np.clip(self._aim, self._aim0 - CAM_AIM_LIMIT, self._aim0 + CAM_AIM_LIMIT)
            self._apply_aim()

        u, w, h, speed = self.ball_in_blade()
        dist = float(np.hypot(u, w))
        centre = 1.0 - min(dist / self.blade_radius, 1.0)
        smooth = self.smooth_penalty
        if self.juggle:  # 1 per step while juggled (the return is the time juggled), light centring, straight hits
            juggled, straight = self._count_bounce(dist)
            reward = float(juggled) + JUGGLE_CENTRE_W * centre + straight
            if release is not None and self.juggle_target is not None and release > RELEASE_MIN_VZ:
                # the hit sets the height: reward leaving the blade at the speed that reaches the commanded height
                v_need = float(np.sqrt(2 * 9.81 * self.target_apex))
                reward += RELEASE_W * (1.0 - min(abs(release - v_need) / RELEASE_SPAN, 1.0))
                self._release_errs.append(release - v_need)
            if impact is not None:  # a clean hit: blade level and moving straight up when the ball lands on it
                tilt, lateral = impact
                reward += HIT_LEVEL_W * float(np.exp(-(tilt / HIT_LEVEL_SIGMA) ** 2))
                reward += HIT_LATERAL_W * float(np.exp(-(lateral / HIT_LATERAL_SIGMA) ** 2))
                self._impacts.append(impact)
            qd = self.d.qvel[self.v_adr].copy()
            in_window = self.ball_lift()[0] < HIT_WINDOW_LIFT
            if in_window:
                if self.juggle_smooth is not None:
                    smooth = self.juggle_smooth  # the hit: one quick move allowed
            else:  # between hits: smooth, and no shaking
                if self.juggle_calm is not None:
                    smooth = self.juggle_calm
                reward -= self.juggle_acc * float(np.sum((qd - self._qd_prev) ** 2))
                self._jit_between.append(float(np.abs(a - self._prev_action).mean()))
            self._qd_prev = qd
        else:  # reward (true ball state): near the centre, slow ball
            reward = centre - 0.05 * speed
        reward -= smooth * float(np.sum((a - self._prev_action) ** 2))
        if self.home_penalty:
            reward -= self.home_penalty * float(np.sum(((self.d.qpos[self.q_adr] - self.home_q) / 0.349) ** 2))
        if self.ball_noise:
            self._bn["ou"] = BN_A * self._bn["ou"] + np.sqrt(1 - BN_A ** 2) * self.rng.normal(0, 1, 2) * BN_OU * self._bn["k"]
            self._noisy_ball()
        self._prev_action = a
        self._hist = np.roll(self._hist, 1, axis=0); self._hist[0] = a
        edge = self.blade_radius - 0.5 * self.ball_radius
        on_blade = h < self.rest_height() + 0.01
        if self.airborne_ok or self.juggle:  # in the air is fine; fell past the blade, landed off it, or flew far away is not
            fell = h < -0.01 or (dist > edge and on_blade) or dist > 0.2 or h > 0.5
        else:
            fell = dist > edge or h < -0.01 or h > 0.08
        bad = bool(self.d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        stalled = self.juggle and not juggled
        terminated = bool(fell or bad or stalled)
        if (fell or bad) and not self.juggle:
            reward -= 5.0  # juggling: a drop and a stall both just end the score (time juggled); an extra drop
            # penalty made catching and holding the ball the safe choice
        truncated = self.t >= self.max_steps
        if self.juggle:
            success = bool(truncated and not terminated)
            extra = dict(stalled=bool(stalled), n_bounces=self.n_bounces, juggle_s=self.t / CONTROL_HZ,
                         apex_mean=float(np.mean(self._apexes)) if self._apexes else float("nan"),
                         apex_offset=float(np.mean(self._apex_offsets)) if self._apex_offsets else float("nan"),
                         apex_std=float(np.std(self._apexes)) if len(self._apexes) > 1 else float("nan"),
                         target_err=float(np.mean(np.abs(self._target_errs))) if self._target_errs else float("nan"),
                         release_err=float(np.mean(np.abs(self._release_errs))) if self._release_errs else float("nan"),
                         hit_tilt_deg=float(np.rad2deg(np.mean([i[0] for i in self._impacts]))) if self._impacts else float("nan"),
                         hit_lateral=float(np.mean([i[1] for i in self._impacts])) if self._impacts else float("nan"),
                         jitter_between=float(np.mean(self._jit_between)) if self._jit_between else float("nan"))
        else:
            success = bool(truncated and not terminated and dist < self.success_radius and on_blade)
            extra = {}
        return self._obs(), float(reward), terminated, truncated, self._info(dist=dist, fell=bool(fell), bad_qacc=bad,
                                                                            is_success=success, **extra)

    def _contact_event(self):
        """Juggling, once per substep: ("impact", (blade tilt from level rad, blade sideways speed m/s)) on the first
        substep of a new ball/blade contact, ("release", ball vertical speed m/s) on the substep the ball leaves the
        blade, else None."""
        d, n = self.d, self.d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        b, p = self.ball_geom, self.blade_geom
        touching = bool(np.any(((g1 == b) & (g2 == p)) | ((g1 == p) & (g2 == b))))
        new, left = touching and not self._touching, self._touching and not touching
        self._touching = touching
        if left:
            return "release", float(d.qvel[self.ball_vadr + 2])
        if not new:
            return None
        normal = d.geom_xmat[p].reshape(3, 3)[:, 2]
        tilt = float(np.arccos(np.clip(abs(normal[2]), 0.0, 1.0)))
        vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.m, d, mujoco.mjtObj.mjOBJ_GEOM, p, vel, 0)
        return "impact", (tilt, float(np.hypot(vel[3], vel[4])))

    def _new_target(self) -> None:
        """Juggling with a height command: a new target apex height, held for 1-3 counted bounces."""
        if self.juggle_target is None:
            self.target_apex = 0.0
            return
        self.target_apex = float(self.rng.uniform(*self.juggle_target))
        self._target_left = int(self.rng.integers(1, 4))

    def _count_bounce(self, dist: float) -> tuple[bool, float]:
        """Juggling: at the top of each flight (vertical velocity turns negative, ball off the blade), count a bounce if
        it peaked >= JUGGLE_MIN_APEX above the resting position. Returns (juggled: last bounce recent, straight-hit
        bonus: JUGGLE_STRAIGHT_W at a counted apex right over the blade centre, falling off with the offset `dist`)."""
        lift, _ = self.ball_lift()
        vz = float(self.d.qvel[self.ball_vadr + 2])
        bonus = 0.0
        if self._vz_prev > 0 >= vz and lift >= JUGGLE_MIN_APEX:
            self.n_bounces += 1
            self._last_bounce = self.t
            self._apexes.append(lift)
            self._apex_offsets.append(dist)
            bonus = JUGGLE_STRAIGHT_W * float(np.exp(-(dist / JUGGLE_STRAIGHT_SIGMA) ** 2))
            if self.juggle_target is not None:  # how close the bounce came to the commanded height
                self._target_errs.append(lift - self.target_apex)
                bonus += self.height_w * (1.0 - min(abs(lift - self.target_apex) / HEIGHT_SPAN, 1.0))
                self._target_left -= 1
                if self._target_left <= 0:
                    self._new_target()
        self._vz_prev = vz
        return self.t - self._last_bounce <= round(JUGGLE_STALL_S * CONTROL_HZ), bonus

    def _render(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.m, *self.image_hw)
        self._renderer.update_scene(self.d, camera=self.cam_id)
        img = self._renderer.render()
        if self.domain_rand:
            c, b, s = self._img_aug
            img = np.clip(img.astype(np.float32) * c + b + self.rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)
        return img

    def _obs(self):
        if self.obs_mode == "controller":
            return self.controller_obs()
        return {"image": self._render()}

    def controller_obs(self):
        """The controller observation (see module docstring). Available in every obs_mode."""
        q = self.d.qpos[self.q_adr] - self.home_q
        qd = self.d.qvel[self.v_adr] * 0.1
        u, w, h, _ = self.ball_in_blade()
        vel = self.d.qvel[self.ball_vadr : self.ball_vadr + 3]
        _, _, uu, ww = self.blade_frame()
        ball = np.array([u / self.blade_radius, w / self.blade_radius, vel @ uu / 0.5, vel @ ww / 0.5])
        if self.ball_noise:
            ball = self._ball_obs
        return np.concatenate([ball, q / 0.2, qd, self._hist.ravel()]).astype(np.float32)

    def _noisy_ball(self):
        """Once per step: noisy ball position + FdVel velocity, cached for controller_obs()."""
        u, w, _, _ = self.ball_in_blade()
        b = self._bn
        p = np.array([u, w]) / self.blade_radius * b["gain"] + b["bias"] + b["ou"] + self.rng.normal(0, 1, 2) * BN_WHITE * b["k"]
        self._ball_obs = self._fd.update(p)

    def rest_height(self) -> float:
        """Ball centre above the blade centre plane when resting on it, m."""
        return float(self.m.geom_size[self.blade_geom][1] + self.ball_radius)

    def ball_lift(self) -> tuple[float, float]:
        """Ball height above its resting position on the blade (m) and velocity along the blade normal (m/s)."""
        _, _, h, _ = self.ball_in_blade()
        n = self.blade_frame()[1]
        return h - self.rest_height(), float(self.d.qvel[self.ball_vadr:self.ball_vadr + 3] @ n)

    def _info(self, **extra):
        u, w, h, speed = self.ball_in_blade()
        return {"ball_u": u, "ball_w": w, "ball_h": h, "ball_speed": speed, "kick_z": self.kick_z_on, "juggle": self.juggle, **extra}

    def since_bounce(self) -> float:
        """Juggling: seconds since the last counted bounce (0 outside juggling episodes)."""
        return (self.t - self._last_bounce) / CONTROL_HZ if self.juggle else 0.0

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
