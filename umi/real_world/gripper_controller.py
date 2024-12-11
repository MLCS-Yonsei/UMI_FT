import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from rtde_io import RTDEIOInterface
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from umi.common.precise_sleep import precise_wait


class GripperCommand(enum.Enum):
    STOP = 0
    SCHEDULE_OPEN = 1
    SCHEDULE_CLOSE = 2

class GripperController(mp.Process):
    def __init__(self,
                 shm_manager: SharedMemoryManager,
                 robot_ip,
                 launch_timeout=3,
                 soft_real_time=False,
                 verbose=False,
                 receive_latency=0.0):
        super().__init__(name="GripperController")
        self.robot_ip = robot_ip
        self.launch_timeout = launch_timeout
        self.soft_real_time = soft_real_time
        self.verbose = verbose
        self.receive_latency = receive_latency

        # Build input queue
        example = {
            'cmd': GripperCommand.SCHEDULE_OPEN.value,
            'target_time': 0.0
        }
        self.input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # Build ring buffer for state (if needed)
        example_state = {
            'gripper_width': 0,  # False for closed, True for open
            'gripper_timestamp': time.time()
        }
        self.ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=100,
            get_time_budget=0.2,
            put_desired_frequency=10  # Adjust as needed
        )

        self.ready_event = mp.Event()

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[GripperController] Process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': GripperCommand.STOP.value,
            'target_time': time.time()
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()

    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()

    def stop_wait(self):
        self.join()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= command methods ============
    def schedule_open(self, target_time: float):
        message = {
            'cmd': GripperCommand.SCHEDULE_OPEN.value,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def schedule_close(self, target_time: float):
        message = {
            'cmd': GripperCommand.SCHEDULE_CLOSE.value,
            'target_time': target_time
        }
        self.input_queue.put(message)

    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k, out=out)

    def get_all_state(self):
        return self.ring_buffer.get_all()

    # ========= main loop in process ============
    def run(self):
        # Enable soft real-time if specified
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))

        # Start RTDEIO interface
        robot_ip = self.robot_ip
        rtde_io = RTDEIOInterface(hostname=robot_ip)

        try:
            if self.verbose:
                print(f"[GripperController] Connected to robot: {robot_ip}")

            keep_running = True
            t_start = time.monotonic()
            iter_idx = 0
            gripper_state = 1 # Assume gripper starts opened

            scheduled_commands = [] # we use digital command which means we can not use interpolation

            self.ready_event.set()

            while keep_running:
                # Fetch commands
                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # Process incoming commands
                for i in range(n_cmd):
                    cmd = commands['cmd'][i]
                    target_time = commands['target_time'][i]

                    if cmd == GripperCommand.STOP.value:
                        keep_running = False
                        break
                    elif cmd == GripperCommand.SCHEDULE_OPEN.value:
                        scheduled_commands.append({
                            'cmd': GripperCommand.OPEN.value,
                            'target_time': target_time
                        })
                    elif cmd == GripperCommand.SCHEDULE_CLOSE.value:
                        scheduled_commands.append({
                            'cmd': GripperCommand.CLOSE.value,
                            'target_time': target_time
                        })
                    else:
                        keep_running = False
                        break
                
                # Execute scheduled commands
                remaining_commands = []
                for command in scheduled_commands:
                    if command['target_time'] <= time.time():
                        # Time to execute the command
                        if command['cmd'] == GripperCommand.OPEN.value:
                            # Open the gripper
                            rtde_io.setToolDigitalOut(0, True)   # Tool DO0 HIGH
                            rtde_io.setToolDigitalOut(1, False)  # Tool DO1 LOW
                            gripper_state = 1
                            if self.verbose:
                                print("[GripperController] Gripper opened.")
                        elif command['cmd'] == GripperCommand.CLOSE.value:
                            # Close the gripper
                            rtde_io.setToolDigitalOut(0, False)  # Tool DO0 LOW
                            rtde_io.setToolDigitalOut(1, True)   # Tool DO1 HIGH
                            gripper_state = 0
                            if self.verbose:
                                print("[GripperController] Gripper closed.")
                    else:
                        # Not yet time to execute
                        remaining_commands.append(command)
                scheduled_commands = remaining_commands

                # Update gripper state (if needed)
                state = {
                    'gripper_width': gripper_state,
                    'gripper_timestamp': time.time() - self.receive_latency
                }
                self.ring_buffer.put(state)

                if iter_idx == 0:
                        self.ready_event.set()

                # Regulate frequency
                dt = 1 / self.frequency
                t_wait_until = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_until, time_func=time.monotonic)
                
                iter_idx += 1


        finally:
            # Disconnect
            rtde_io.disconnect()
            if self.verbose:
                print(f"[GripperController] Disconnected from robot: {robot_ip}")