# Cart-Pole MuJoCo Playground

Hands-on experiments with MuJoCo physics simulation using a cart-pole model.

## Setup

Requires Python 3.12 with MuJoCo installed:

```sh
pip install mujoco stable-baselines3 gymnasium[mujoco]
```

## Running the Viewer

```sh
python view_cart_pole.py
```

Or directly:

```sh
python -m mujoco.viewer --mjcf=cart_pole.xml
```

## Viewer Controls (Mac Trackpad)

### Navigation
- **One-finger drag**: Rotate camera
- **Two-finger drag**: Pan camera
- **Pinch/scroll**: Zoom in/out

### Simulation Control
- **Space**: Toggle play/pause
- **Backspace** (Delete key): Reset simulation
- **Reload button**: Reload XML file fresh

### Applying Forces (Perturbations)
1. **Double-click** on a body to select it (body gets highlighted)
2. **Ctrl + two-finger drag**: Apply force to selected body
3. **Release**: Force stops, physics continues

When dragging with Ctrl held, you'll see **two bodies**:
- The **actual body** (connected to physics)
- A **ghost/target body** showing where you're dragging to

The gap between them represents the force being applied. MuJoCo applies forces to push the real body toward the ghost. This is the perturbation visualization - not an arrow, but a target indicator.

### Selection
- **Double-click**: Select a body
- **Click empty space**: Deselect (may not always work in Python viewer)
- **Esc**: Should deselect (limited in Python viewer)

### Actuator Control
Expand the **Control** panel on the right side:
- `cart_motor` slider: Apply force to cart (-10 to +10 N)
- Drag slider to push cart left/right, which tips the pole

## Model Details

From `cart_pole.xml`:
- **Cart**: Blue box, 1 kg, slides on rail (x-axis)
- **Pole**: Orange-yellow rod (`rgba="1 0.75 0.1 1"`), 0.25 kg, hinges at base
- **Pole tip**: Red sphere (`rgba="0.95 0.20 0.12 1"`), 0.04 kg
- **Gravity**: Earth standard (9.81 m/s^2)
- **No balancing controller**: Pole will fall when disturbed

## Notes

- The Python viewer (`mujoco.viewer.launch`) has fewer keyboard shortcuts than the standalone C++ `simulate` app
- Keys like P, F, R for visualization toggles may not work in Python viewer
- The pole color is intentionally orange (not yellow) - defined in the XML
- The pole starts perfectly upright; it may stay balanced until disturbed due to numerical precision, not because it can balance itself

## Scripts

### `view_cart_pole.py`
Basic viewer - no controller, pole falls when pushed.

### `balance.py`
PD controller that actively balances the pole. Run with:
```sh
python balance.py
```
Push the pole with Ctrl+drag - it recovers!

### `benchmark.py`
Measures simulation speed on your machine:
```sh
python benchmark.py
```

### `train_rl.py`
Trains a PPO agent on our custom cart-pole environment:
```sh
python train_rl.py
```
Saves checkpoints to `checkpoints/` directory.

### `watch_rl.py`
Visualize a trained RL policy:
```sh
python watch_rl.py checkpoints/cartpole_ppo_final.zip
```
Push the pole with Ctrl+drag - the learned policy recovers!

### `train_rl_gym.py`
Train on Gymnasium's built-in InvertedPendulum-v5 (for comparison):
```sh
python train_rl_gym.py
```

### `watch_gym.py`
Visualize the Gymnasium-trained policy:
```sh
python watch_gym.py
```

## Concepts Learned

### Timesteps
MuJoCo computes physics in discrete steps (0.002s each = 500 steps/sec). Each step: read state → compute forces → update positions.

### PD Control
Classical feedback controller:
```python
force = Kp * pole_angle + Kd * pole_velocity
```
- **Kp (proportional)**: React to current error
- **Kd (derivative)**: React to rate of change (damping)

Tuned values: Kp=300, Kd=50 with motor limit ±100N.

### Why RL for complex robots?
PD works for simple systems (2 joints). Complex robots (20+ joints) have too many interacting parameters. RL learns the control policy automatically through trial and error.

### RL Training Lessons Learned

**Frame skip is critical.** Our XML uses `timestep=0.002` (500 Hz physics), but Gymnasium's InvertedPendulum uses `timestep=0.02` with `frame_skip=2` (25 Hz control). Without frame skip, the policy makes 500 decisions per second - each with tiny effect, making learning extremely hard.

| Setting | Physics Hz | Control Hz | Result |
|---------|-----------|------------|--------|
| No frame skip | 500 | 500 | Failed - cart drifts to edge |
| frame_skip=20 | 500 | 25 | Works - stable balancing |

**Simple rewards work.** Complex reward shaping (penalizing angle, position, velocity, action) didn't help. Gymnasium just uses `+1 per step survived` and it works fine.

**Training scale.** 50k steps = failure. 500k steps = success. RL needs way more samples than you'd expect.

## Next Steps

1. ~~Add a simple feedback controller to balance the pole~~ ✓
2. ~~Train an RL policy for balancing~~ ✓
3. Compare learned policy vs hand-tuned PD

## Sim2Real Reference: Rotary Inverted Pendulum

Studying [ferrolho/rotary-inverted-pendulum](https://github.com/ferrolho/rotary-inverted-pendulum) - a complete sim2real project using MuJoCo + SB3 + Arduino.

### Key Learnings

**Observations vs Actions**
- Observations = inputs to neural network (what it sees)
- Actions = outputs from neural network (what it does)
- MuJoCo knows everything; you choose what to feed to RL

**Ferrolho's observation vector (6 values per frame):**
1. `motor_pos` - arm rotation angle (rad)
2. `sin(θ)` - pendulum angle from upright
3. `cos(θ)` - pendulum angle from upright
4. `motor_vel` - arm angular velocity (rad/s)
5. `pendulum_vel` - pendulum angular velocity (rad/s)
6. `prev_action` - last motor command sent

**Why sin/cos instead of raw angle?**
Raw angle wraps at ±180° (jumps from +179° to -179°). sin/cos are smooth everywhere - small physical change = small numerical change.

**Why prev_action?**
Real motors have lag. Knowing what you commanded helps the network anticipate where motor is heading, not just where it is now. Prevents overshoot.

**Frame stacking (K frames)**
Feed last K snapshots instead of just current one. If K=4 with 6 values/frame = 24 total inputs.

When you need it:
- Velocity not measured (must compute from position changes)
- Sensors are noisy (need to average/filter)
- Hidden state like motor lag or sensor bias

When 1 frame is enough:
- Clean observations with position + velocity
- No hidden delays or drift
- Simulation with perfect sensor data (like our cart-pole)

**Action space**
Single number [-1 to +1] = motor direction + intensity. Gets scaled to real units (e.g., ×100 for Newtons).

**Domain randomization**
Make the simulation intentionally worse to match real-world imperfections:
- Add Gaussian noise to sensor readings
- Randomize physics parameters (friction, mass)
- Quantize encoder readings to match hardware resolution

Goal: Policy trained on "crappy" sim transfers better to real hardware.

**Reward function design**
Ferrolho uses "Quanser quadratic-cost form" - borrowed from classical control theory literature, not invented:
- Quadratic penalties: θ², velocity², action² - big errors hurt exponentially more
- Alive bonus: +1 per step survived - prevents "crash early to stop accumulating penalty"
- Standard formula for Furuta pendulum control, dating back decades

### Our cart-pole vs Ferrolho's pendulum

| Aspect | Our cart-pole | Ferrolho pendulum |
|--------|---------------|-------------------|
| Observations | 4 | 6 (+ frame stacking) |
| sin/cos angle | No (stays <30°) | Yes (full rotation) |
| prev_action | No (sim only) | Yes (real hardware) |
| Frame stacking | No | Yes (K=1-4) |
| Action | 1 (cart force) | 1 (motor torque) |

## Session Log

### 2024-09-28: Initial exploration
- Learned basic viewer navigation (camera rotation, pan, zoom)
- Figured out force application: Ctrl + trackpad drag on selected body
- Discovered perturbation visualization shows as ghost target body, not arrow
- Confirmed pole color is orange by design (XML rgba values)
- Reset vs Reload: Reset restores state, Reload re-reads XML file

### 2024-09-29: PD Controller
- Implemented PD balancing controller (`balance.py`)
- Learned control loop: read sensors → compute force → apply → repeat 500x/sec
- Initial sign was wrong (pushed cart away from lean) - fixed
- Increased motor limit (10N → 100N) and gains (Kp=300, Kd=50) for stronger recovery
- Benchmarked: ~300,000 steps/sec on MacBook (592x realtime)
- Discussed why classical control works for cart-pole but RL needed for complex robots

### 2024-09-29: RL Training
- First attempt: 50k steps with no frame skip - policy oscillated wildly, cart drifted to edge
- Compared with Gymnasium's InvertedPendulum-v5 which trained successfully
- Key insight: Gymnasium uses frame_skip (control at 25 Hz, not 500 Hz)
- Added frame_skip=20 to match Gymnasium's control frequency
- Simplified reward to just +1 per step (like Gymnasium)
- 500k steps: ep_len_mean reached 500 (full episodes)
- Final policy balances stably - can push pole and it recovers!
