# Assets

- **SO-101 arm** (`assets/*.stl`, and the arm part of `so101_new_calib_paddle.xml`): from
  [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (`Simulation/SO101`), Apache-2.0, see `LICENSE`.
  Modified here (`so101_new_calib_paddle.xml`, from upstream `so101_new_calib.xml`): joint ranges widened to match a real arm's calibration
  (originals listed in the file header), white printed parts, the upstream wrist-camera mount + a `<camera name="wrist">` matched to a real
  wrist-camera frame by eye, a table-tennis paddle clamped in the gripper (visual meshes + a flat cylinder collider, 90 g), and only the
  meshes this scene needs.
  Motor parameters originally adapted upstream from the [Open Duck Mini](https://github.com/apirrone/Open_Duck_Mini) project.
- **Paddle** (`assets/paddle_Material_00{4,5,6}.obj`): MIT, see `assets/PADDLE_LICENSE.txt`.
- `scene_balance.xml`: floor, lights and the ping-pong ball (40 mm, 2.7 g).
