# Passive hinge versus powered servo

This is a small MuJoCo experiment containing two mechanically identical elbow
joints:

- **Orange (left):** passive hinge. Gravity and damping make it hang downward.
- **Blue (right):** the same hinge with a position actuator. Its default target
  is `0 rad`, so it actively holds the forearm horizontally.

The joint limits are +/-120 degrees for both elbows. The powered target can be
changed by expanding MuJoCo's **Control** panel and moving the
`powered_servo` slider.

Run on macOS with:

```sh
uv run mjpython /Users/adityasudhakar/mujoco/elbow-demo/view_elbows.py
```

Press **Space** or click **Run** if the simulation opens paused.

## Key Learnings

**How MuJoCo knows the difference between passive and powered joints:**

The names `passive_elbow` and `powered_elbow` are just human-readable labels - MuJoCo doesn't care about the English meaning. What makes a joint "powered" is connecting it to an actuator:

```xml
<actuator>
  <position joint="powered_elbow" kp="8" kv="0.25"/>
</actuator>
```

- `passive_elbow` → no actuator references it → moves freely under gravity
- `powered_elbow` → referenced by actuator → torque applied to reach target

**Servo gains explained:**

- `kp` (position gain): How hard the servo pushes when joint is away from target
- `kv` (velocity gain): Damping to prevent overshoot and bouncing

Formula: `motor force = kp × angle_error − kv × joint_speed`

**What determines resting position:**

MuJoCo calculates rest from: gravity, mass/inertia, joint axis/limits, damping, collisions, and (for powered joints) servo target + strength.

**A joint with an actuator is still movable** - the actuator just applies torque toward a target. Strong external force can still move it, especially if actuator force is limited.

**Why the powered forearm rests horizontally:**

The forearm geometry is drawn along the X axis:

```xml
<geom type="capsule" fromto="0 0 0 0.27 0 0"/>
```

That is its zero-angle pose. MuJoCo initializes actuator controls to zero, so
the position actuator holds the joint at `0 rad`, which is horizontal in this
model. Pointing it straight up requires a target near `-1.57 rad` (-90 deg).

**What "powered" means here:**

The position actuator can continuously apply up to the configured torque limit.
MuJoCo does not automatically simulate battery charge, voltage sag, electrical
power use, servo heating, or shutdown; those require additional modeling.
