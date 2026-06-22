# Deployment environment

A single conda env combining the MuJoCo replay/IK stack (`mujoco`, `mink`,
`viser`) with `ur_rtde` for driving the real UR3e arms — everything both stages
(`mujoco_replay/` and `robot_replay/`) need.

```bash
conda env create -f env/deployment.yml
conda activate deployment
```

Notes:
- `env/deployment.yml` is a full `conda env export` of a known-working env
  (Python 3.12.13), treated mainly as a version reference. Regenerate it on a
  known-good machine with:
  ```bash
  conda env export -n deployment | grep -v '^prefix:' > env/deployment.yml
  ```
- The Sharpa Wave hand SDK is proprietary and **not** shipped here. Drop it into
  `robot_replay/Sharpa/` (the path in `robot_replay/config.example.yaml`).
- `mjviser` is vendored under `mujoco_replay/mjviser/`, so it is not a pip
  dependency.
