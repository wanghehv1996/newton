
import numpy as np

class TrajectoryAnimation:
    def __init__(self):
        pass

    def get_pose(self, key, time):
        return [], [] # transform and state


class KeyFrameTrajectoryAnimation:
    def __init__(self):
        self.effector_trajectories = {}

    @staticmethod
    def _quat_normalize(q):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n == 0.0:
            return q
        return (q / n).astype(np.float64)

    @staticmethod
    def _quat_slerp(q0, q1, t):
        q0 = KeyFrameTrajectoryAnimation._quat_normalize(q0)
        q1 = KeyFrameTrajectoryAnimation._quat_normalize(q1)

        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        # If the quaternions are close, use linear interpolation
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return KeyFrameTrajectoryAnimation._quat_normalize(q)

        theta_0 = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta_0 = np.sin(theta_0)
        theta = theta_0 * t
        sin_theta = np.sin(theta)

        s0 = np.sin(theta_0 - theta) / sin_theta_0
        s1 = sin_theta / sin_theta_0
        return (s0 * q0 + s1 * q1).astype(np.float64)

    def resample(self, fps: float = 60.0):
        """Resample trajectories uniformly in time at a fixed frame rate.

        - Each second is split into `fps` frames (dt = 1/fps).
        - Positions are linearly interpolated.
        - Rotations (quaternions) use SLERP on [qx, qy, qz, qw].
        - States are linearly interpolated.
        """
        dt = 1.0 / float(fps)
        for key, traj in list(self.effector_trajectories.items()):
            times = np.asarray(traj["times"], dtype=float)
            transforms = np.asarray(traj["transforms"], dtype=float)
            states = np.asarray(traj["states"], dtype=float)

            start_t = float(times[0])
            end_t = float(times[-1])

            # Build uniform timeline; exclude the final time for interpolation loop
            uniform_times = list(np.arange(start_t, end_t, dt))

            new_times = []
            new_transforms = []
            new_states = []

            for t in uniform_times:
                # find bracketing keyframe indices
                j = int(np.searchsorted(times, t, side="right") - 1)
                j = max(0, min(j, len(times) - 2))

                t0 = float(times[j])
                t1 = float(times[j + 1])
                a = 0.0 if t1 == t0 else (float(t) - t0) / (t1 - t0)
                a = float(np.clip(a, 0.0, 1.0))

                p0 = transforms[j, 0:3]
                q0 = transforms[j, 3:7]
                s0 = float(states[j])

                p1 = transforms[j + 1, 0:3]
                q1 = transforms[j + 1, 3:7]
                s1 = float(states[j + 1])

                pos = (1.0 - a) * p0 + a * p1
                quat = KeyFrameTrajectoryAnimation._quat_slerp(q0, q1, a)
                # Asymmetric rule for discrete state:
                # - 0 -> 1: switch immediately to 1 at the start of the segment
                # - 1 -> 0: interpolate linearly
                # - equal: keep constant
                if s0 < s1:
                    state = (1.0 - a) * s0 + a * s1
                elif s0 > s1:
                    state = (1.0 - a) * s0 + a * s1
                else:
                    state = s0

                new_times.append(float(t))
                new_transforms.append(np.concatenate([pos, quat], axis=0))
                new_states.append(state)

            # Append the very last keyframe exactly
            new_times.append(end_t)
            new_transforms.append(transforms[-1])
            new_states.append(float(states[-1]))

            self.effector_trajectories[key] = {
                "times": np.asarray(new_times, dtype=float),
                "transforms": np.asarray(new_transforms, dtype=float),
                "states": np.asarray(new_states, dtype=float),
            }

    def init_lift2_folding(self):
        # init trajectory
        self.effector_trajectories["left_gripper"]={
            # "times": np.array([0.0, 1.0, 2.0, 4.5, 12.5, 15.0, 15.5, 20.0]),
            "times": np.array([0.0, 1.0, 2.0, 3.0, 6.0, 7.0, 8.0]),
            "transforms":np.array([
                [0.26322999596595764, 0.24842043220996857, 0.6, 0.0, 0.0, 0.0, 1.0], # rest
                [0.385, 0.23, 0.48, 0.0, 0.2, 0.0, 0.9797], # fetch
                [0.385, 0.23, 0.48, 0.0, 0.2, 0.0, 0.9797], # grasp
                [0.385, 0.23, 0.6, 0.0, 0.2, 0.0, 0.9797], # lift
                [0.65, -0.09, 0.6, 0.0, 0.2, 0.0, 0.9797], # move
                # [0.65, -0.09, 0.55, 0.0, 0.2, 0.0, 0.9797], # down
                [0.65, -0.09, 0.6, 0.0, 0.2, 0.0, 0.9797], # release
                [0.385, 0.23, 0.6, 0.0, 0.2, 0.0, 0.9797], #return
            ]),
            # "states":np.array([1, 0, 0, 0, 0, 0, 1, 1]),
            "states":np.array([1, 1, 0, 0, 0, 1, 1]),
        }
        self.effector_trajectories["right_gripper"]={
            "times": np.array([0.0, 1.0, 5.0, 6.0, 7.0, 8.0,11.0, 12.0, 13.0]),
            "transforms":np.array([
                [0.26322999596595764, -0.24841956794261932, 0.6, 0.0, 0.0, 0.0, 1.0], # rest
                [0.385, -0.23, 0.6, 0.0, 0.3, 0.0, 0.95393920141], # rest
                [0.385, -0.23, 0.6, 0.0, 0.3, 0.0, 0.95393920141], # rest
                [0.385, -0.23, 0.48, 0.0, 0.2, 0.0, 0.9797], # fetch
                [0.385, -0.23, 0.48, 0.0, 0.2, 0.0, 0.9797], # grasp
                [0.385, -0.23, 0.6, 0.0, 0.2, 0.0, 0.9797], # lift
                [0.65, 0.09, 0.6, 0.0, 0.2, 0.0, 0.9797], # move
                # [0.65, 0.09, 0.55, 0.0, 0.2, 0.0, 0.9797], # down
                [0.65, 0.09, 0.6, 0.0, 0.2, 0.0, 0.9797], # release
                [0.385, -0.23, 0.6, 0.0, 0.2, 0.0, 0.9797], #return
            ]),
            # "states":np.array([1, 1, 1, 0, 0, 0, 0, 0, 1, 1]),
            "states":np.array([1, 1, 1, 1, 0, 0, 0, 1, 1]),
        }

        # Densify keyframes for smoother playback (60 FPS)
        self.resample(fps=30.0)

    def get_pose(self, key, time):

        if self.effector_trajectories[key] is not None:
            times = self.effector_trajectories[key]["times"]
            # time_mod = (
            #     time
            #     if time < times[-1]
            #     else time % times[-1]
            # )
            time_mod = min(time, times[-1])

            time_id_upper = np.searchsorted(times, time_mod)
            time_id_lower = time_id_upper-1

            if time_id_upper == 0:
                return self.effector_trajectories[key]["transforms"][0], self.effector_trajectories[key]["states"][0]
            else:
                transforms = self.effector_trajectories[key]["transforms"]
                states = self.effector_trajectories[key]["states"]

                # interp transform
                time_upper = times[time_id_upper]
                time_lower = times[time_id_lower]
                theta = (time - time_lower)/(time_upper - time_lower)
                theta = min(1, max(0, theta)) # clamp to [0,1]
                target = transforms[time_id_upper] * theta + transforms[time_id_lower] * (1-theta)

                # Asymmetric rule for discrete state in query:
                # - 0 -> 1: switch immediately to 1 at the start of the segment
                # - 1 -> 0: interpolate linearly
                # - equal: keep constant
                s0 = float(states[time_id_lower])
                s1 = float(states[time_id_upper])
                if s0 < s1:
                    state = s1
                elif s0 > s1:
                    state = (1.0 - theta) * s0 + theta * s1
                else:
                    state = s0
                return target, state
        else:
            print(f"find no effector trajectories for {key}")
            return transforms[0]*0, states[0]*0


# Simple test
# anim = KeyFrameTrajectoryAnimation()
# anim.init_lift2_folding()
# for i in range(0, 20):
#     pose = anim.get_pose("left_gripper", i/10)
#     print(f"time = {i/10}, {pose}")

