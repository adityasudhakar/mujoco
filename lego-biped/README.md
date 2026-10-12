# Planar LEGO biped

This is a deliberately small bridge between the rotary pendulum and a fully
3-D walking robot.

## Physical idea

- Standard LEGO/Technic dimensions are used as the design grid.
- Four actuators: left/right hip and left/right knee.
- Fixed ankles and broad feet in version 1.
- The torso is constrained to the sagittal plane: it may move forward/up and
  rotate forward/backward, but cannot fall sideways or yaw.
- Primitive collision boxes and capsules come first.  Decorative meshes can be
  added after the dynamics work.

Nominal LEGO dimensions used here:

- stud pitch: 8 mm
- plate height: 3.2 mm
- brick height: 9.6 mm
- Technic-hole pitch: 8 mm

The current masses are explicit placeholders.  Before building a real robot,
we will weigh the completed torso, each thigh, each shin, and each foot and
replace them.

Current geometry revision:

- each foot is 240 mm long (twice the original length) and centred on the leg,
  extending 120 mm forward and 120 mm backward;
- the torso is 288 mm tall (twice the original height), with its lower edge and
  hip location unchanged;
- the mass values have deliberately not been doubled; they remain placeholders
  until the physical parts are weighed.

## Milestone 1: scripted motions (no RL)

The acceptance tests are:

1. `walk`: move forward at least 20 cm, finish upright, and remain standing
   for two seconds;
2. `getup_prone`: begin face-down, finish upright, and remain standing for two
   seconds;
3. `getup_supine`: begin face-up, finish upright, and remain standing for two
   seconds.

Current status: the XML model loads and the unperturbed robot stands stably.
After iterative manual tuning, `getup_prone_1` now gets the robot from its
face-down reset to standing.  The walking and supine get-up motions remain
unfinished; the first walking draft moves forward but falls at the end.

[Watch the first successful `getup_prone_1` recording](assets/getup-prone-1-success.mov)
(recorded October 12, 2026).

Run a motion in the interactive viewer:

```bash
/Users/adityasudhakar/toy-cleaner/.venv312/bin/mjpython \
  lego-biped/scripted_motions.py walk
```

Add `--headless` for a numerical check without opening the viewer.

The scripted controller sends joint-position targets at 50 Hz.  MuJoCo runs
the underlying physics at 1 kHz between those commands.

## PPO get-up experiment

`biped_env.py` exposes the four joint-position targets as incremental actions
at 50 Hz.  Each RL step advances 20 MuJoCo physics steps.  Training uses a
curriculum that moves resets from near-standing poses toward the exact prone
keyframe; evaluation always begins at the exact prone keyframe.

Run the environment/API check:

```bash
/Users/adityasudhakar/toy-cleaner/.venv312/bin/python \
  lego-biped/biped_env.py
```

Train 300,000 PPO timesteps:

```bash
/Users/adityasudhakar/toy-cleaner/.venv312/bin/python \
  lego-biped/train_rl.py --timesteps 300000 --run-name getup_300k_v2
```

Watch the best saved policy (`P` pauses and `R` resets):

```bash
/Users/adityasudhakar/toy-cleaner/.venv312/bin/mjpython \
  lego-biped/watch_rl.py
```

Current result (seed 42, 300,000 steps): **failed**.  The best
deterministic policy raised the torso from prone to about 0.174 m and reached
an upright orientation, but it did not reach the 0.28 m height threshold or
hold a full stand.  It scored 0/5 deterministic and 0/50 stochastic prone
successes.  Checkpoints and evaluation logs are in
`checkpoints/getup_300k_v2/`.

## Get-up status and next steps

The initial manual motion-table attempts proved very difficult, and the PPO
experiments above did not work.  Further manual iteration ultimately produced
the successful `getup_prone_1` routine recorded above.  The saved PPO
checkpoints remain experiment records, not working get-up policies.

Next, adapt the approach in
[UTRA RoboSoccer's Unified Humanoid Get-Up](https://github.com/utra-robosoccer/unified-humanoid-getup)
to this simpler four-joint robot:

1. Generate a library of a few thousand physically settled fallen poses by
   randomizing the joints and torso pitch, dropping the robot, and simulating
   until it rests.  Train from these states instead of interpolated poses.
2. Tie the left/right actions together during get-up training, reducing the
   policy to one hip command and one knee command.
3. Observe joint angles and velocities, current servo targets, torso pitch and
   angular velocity, and the previous action.
4. Use actions as bounded rates of change of the servo targets and interpolate
   them between policy updates for smooth motion.
5. Reward torso height first; activate the upright-pitch reward only after a
   useful height is reached.  Also penalize joint speed, abrupt action changes,
   and self-collision.  Keep the existing height-plus-hold acceptance test.
6. First solve nominal prone and supine recovery.  Only then randomize friction,
   mass, centre of mass, damping, actuator strength, pushes, and sensor delay.
7. Replace placeholder masses and actuators with measurements from the LEGO
   build.  After planar sim-to-real works, consider removing the sideways
   constraint and adding actuated ankles.
