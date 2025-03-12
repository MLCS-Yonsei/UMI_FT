from typing import Union
import numbers
import numpy as np
import scipy.interpolate as si

class JointTrajectoryInterpolator:
    def __init__(self, times: np.ndarray, joints: np.ndarray):
        assert len(times) >= 1, "At least one timestamp is required."
        assert len(joints) == len(times), "Number of joint entries must equal number of timestamps."
        if not isinstance(times, np.ndarray):
            times = np.array(times)
        if not isinstance(joints, np.ndarray):
            joints = np.array(joints)

        if len(times) == 1:
            # Special handling for a single timestep.
            self.single_step = True
            self._times = times
            self._joints = joints
        else:
            self.single_step = False
            # Ensure times are sorted.
            assert np.all(times[1:] >= times[:-1]), "Times must be sorted in increasing order."
            self._times = times
            # Create a linear interpolation function along axis 0 (time).
            self.q_interp = si.interp1d(times, joints, 
                                            axis=0, 
                                            assume_sorted=True)
    
    @property
    def times(self) -> np.ndarray:
        return self._times

    @property
    def joints(self) -> np.ndarray:
        if self.single_step:
            return self._joints
        else:
            return self.q_interp(self._times)
    
    def trim(self, start_t: float, end_t: float) -> "JointTrajectoryInterpolator":
        assert start_t <= end_t, "start_t must be less than or equal to end_t."
        times = self.times
        # Keep only those times strictly between start_t and end_t.
        should_keep = (start_t < times) & (times < end_t)
        keep_times = times[should_keep]
        # Include the endpoints.
        all_times = np.concatenate([[start_t], keep_times, [end_t]])
        # Remove duplicates; interp1d requires strictly increasing times.
        all_times = np.unique(all_times)
        # Interpolate joint values at these times.
        all_joints = self(all_times)
        return JointTrajectoryInterpolator(times=all_times, joints=all_joints)
    
    def drive_to_waypoint(self, 
                          joint_target: np.ndarray, 
                          time: float, 
                          curr_time: float,
                          max_joint_speed: float = np.inf
                         ) -> "JointTrajectoryInterpolator":
        assert max_joint_speed > 0, "max_joint_speed must be positive."
        # Ensure the waypoint time is not earlier than the current time.
        time = max(time, curr_time)
        curr_joint = self(curr_time)
        joint_dist = np.linalg.norm(joint_target - curr_joint)
        min_duration = joint_dist / max_joint_speed
        duration = time - curr_time
        # Ensure we allow at least the minimum duration based on speed.
        duration = max(duration, min_duration)
        assert duration >= 0, "Computed duration must be non-negative."
        last_waypoint_time = curr_time + duration

        # Trim trajectory to current time (i.e. remove future waypoints).
        trimmed_interp = self.trim(curr_time, curr_time)
        # Append the new waypoint.
        times = np.append(trimmed_interp.times, [last_waypoint_time])
        joints = np.append(trimmed_interp.joints, [joint_target], axis=0)
        return JointTrajectoryInterpolator(times, joints)

    def schedule_waypoint(self,
                          joint_target: np.ndarray, 
                          time: float, 
                          max_joint_speed: float = np.inf,
                          curr_time: Union[float, None] = None,
                          last_waypoint_time: Union[float, None] = None
                         ) -> "JointTrajectoryInterpolator":
        assert max_joint_speed > 0, "max_joint_speed must be positive."
        if last_waypoint_time is not None:
            assert curr_time is not None, "curr_time must be provided if last_waypoint_time is set."

        start_time = self.times[0]
        end_time = self.times[-1]
        assert start_time <= end_time, "Invalid trajectory times."

        if curr_time is not None:
            if time <= curr_time:
                # Inserting a waypoint earlier than current time has no effect.
                return self
            start_time = max(curr_time, start_time)
            if last_waypoint_time is not None:
                if time <= last_waypoint_time:
                    end_time = curr_time
                else:
                    end_time = max(last_waypoint_time, curr_time)
            else:
                end_time = curr_time

        end_time = min(end_time, time)
        start_time = min(start_time, end_time)
        assert start_time <= end_time, "start_time must be <= end_time."
        assert end_time <= time, "end_time must be <= time."
        if last_waypoint_time is not None:
            if time <= last_waypoint_time:
                assert end_time == curr_time
            else:
                assert end_time == max(last_waypoint_time, curr_time)
        if curr_time is not None:
            assert curr_time <= start_time
            assert curr_time <= time

        # Trim the trajectory to the interval [start_time, end_time].
        trimmed_interp = self.trim(start_time, end_time)
        duration = time - end_time
        end_joint = trimmed_interp(end_time)
        joint_dist = np.linalg.norm(joint_target - end_joint)
        min_duration = joint_dist / max_joint_speed
        duration = max(duration, min_duration)
        assert duration >= 0, "Computed duration must be non-negative."
        last_waypoint_time_new = end_time + duration

        # Append the new waypoint.
        times = np.append(trimmed_interp.times, [last_waypoint_time_new])
        joints = np.append(trimmed_interp.joints, [joint_target], axis=0)
        return JointTrajectoryInterpolator(times, joints)

    def __call__(self, t: Union[numbers.Number, np.ndarray]) -> np.ndarray:

        is_single = False
        if isinstance(t, numbers.Number):
            is_single = True
            t = np.array([t])
        
        if self.single_step:
            # For a single step, simply return the constant joint configuration.
            joints = np.tile(self._joints[0], (len(t), 1))
        else:
            start_time = self.times[0]
            end_time = self.times[-1]
            t = np.clip(t, start_time, end_time)
            joints = self.q_interp(t)
        
        if is_single:
            return joints[0]
        return joints
