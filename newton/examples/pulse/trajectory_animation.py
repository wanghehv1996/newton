
import numpy as np

class TrajectoryAnimation:
    def __init__(self):
        pass

    def get_pose(self, key, time):
        return [], [] # transform and state


class KeyFrameTrajectoryAnimation:
    def __init__(self):
        self.effector_trajectories = {}

    def init_lift2_folding(self):
        # init trajectory
        self.effector_trajectories["left_gripper"]={
            "times": np.array([0.0, 1.0, 2.0, 4.5, 12.5, 15.0, 15.5, 100.0]),
            "transforms":np.array([
                [0.26322999596595764, 0.24842043220996857, 0.5674999952316284, 0.0, 0.0, 0.0, 1.0],# rest
                [0.385, 0.23, 0.48, 0.0, 0.2, 0.0, 0.9797], # fetch
                [0.385, 0.23, 0.48, 0.0, 0.2, 0.0, 0.9797], # grasp
                [0.385, 0.23, 0.6, 0.0, 0.2, 0.0, 0.9797], # lift
                [0.65, -0.09, 0.6, 0.0, 0.2, 0.0, 0.9797], # move
                [0.65, -0.09, 0.55, 0.0, 0.2, 0.0, 0.9797], # down
                [0.65, -0.09, 0.55, 0.0, 0.2, 0.0, 0.9797], # release
                # [0.35, 0.22, 0.5, 0.0, 0.3, 0.0, 0.95393920141], # fetch
                # [0.35, 0.22, 0.5, 0.0, 0.3, 0.0, 0.95393920141], # grasp
                # [0.35, 0.22, 0.6, 0.0, 0.3, 0.0, 0.95393920141], # lift
                # [0.65, -0.09, 0.55, 0.0, 0.0, 0.0, 1.0], # move
                # [0.65, -0.09, 0.55, 0.0, 0.0, 0.0, 1.0],
                [0.26322999596595764, 0.24842043220996857, 0.5674999952316284, 0.0, 0.0, 0.0, 1.0], #return
            ]),
            "states":np.array([1, 0, 0, 0, 0, 0, 1, 1]),
        }
        self.effector_trajectories["right_gripper"]={
            "times": np.array([0.0, 1.0]),
            "transforms":np.array([
                [0.26322999596595764, -0.24841956794261932, 0.5674999952316284, 0.0, 0.0, 0.0, 1.0],
                [0.35, -0.25, 0.5, 0.0, 0.3, 0.0, 0.95393920141],
            ]),
            "states":np.array([1, 0]),
        }

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
                
                # take lower state 
                state = states[time_id_lower]
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

