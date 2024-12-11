import os
import time
import enum
import socket
import struct
import numpy as np
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer


class Command(enum.Enum):
    SHUTDOWN = 0

class FTSensorController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            sensor_ip,
            pc_ip,
            udp_port=8890,
            frequency=1000,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.0,
            verbose=False
            ):
        super().__init__(name="FTSensorController")
        self.sensor_ip = sensor_ip
        self.pc_ip = pc_ip
        self.udp_port = udp_port
        self.frequency = frequency
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SHUTDOWN.value,
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example_state = {
            'force': np.zeros((3,), dtype=np.float64),
            'torque': np.zeros((3,), dtype=np.float64),
            'ft_receive_timestamp': time.time(),
            'ft_timestamp': time.time()
        }

        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example_state,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )
        
        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[WSGController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
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
        
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        # start connection
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.bind((self.pc_ip, self.udp_port))

            # send bias command
            bias_comm = bytearray.fromhex("00 03 04")
            sock.sendto(bias_comm, (self.sensor_ip, self.udp_port))
            if self.verbose:
                print("[FTSensorController] Sent bias command to FT sensor.")
            
            transmit_comm = bytearray.fromhex("00 03 02")
            sock.sendto(transmit_comm, (self.sensor_ip, self.udp_port))
            if self.verbose:
                print("[FTSensorController] Sent transmit command to FT sensor.")
            
            keep_running = True

            t_start = time.monotonic()
            iter_idx = 0

            self.ready_event.set()

            while keep_running:
                t_now = time.monotonic()
                dt = 1 / self.frequency

                try:
                    data, addr = sock.recvfrom(1024)
                    t_receive = time.time()
                    if len(data) == 52:
                        unpacked_data = struct.unpack('>13f', data)
                        Fx, Fy, Fz, Tx, Ty, Tz, Ax, Ay, Az, Gx, Gy, Gz, Temp = unpacked_data
                        force = np.array([Fx, Fy, Fz])
                        torque = np.array([Tx, Ty, Tz])
                        # Build state dictionary
                        state = {
                            'force': force,
                            'torque': torque,
                            'ft_receive_timestamp': t_receive,
                            'ft_timestamp': t_receive - self.receive_latency
                        }
                        print("FT sensor state: ", state)
                        self.ring_buffer.put(state)
                    else:
                        if self.verbose:
                            print("[FTSensorController] Received data length is incorrect.")

                except socket.timeout:
                    if self.verbose:
                        print("[FTSensorController] Socket timeout, no data received.")

                except Exception as e:
                    if self.verbose:
                        print(f"[FTSensorController] Exception occurred: {e}")

                try:
                    commands = self.input_queue.get_all()
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0
                

                for i in range(n_cmd):
                    cmd = commands['cmd'][i]

                    if cmd == Command.SHUTDOWN.value:
                        keep_running = False
                        break
                    else:
                        if self.verbose:
                            print(f"[FTSensorController] Unknown command: {cmd}")
                
                # Regulate frequency
                t_wait_until = t_start + (iter_idx + 1) * dt
                time_to_wait = t_wait_until - time.monotonic()
                if time_to_wait > 0:
                    time.sleep(time_to_wait)
                iter_idx += 1

        finally:
            sock.close()
            self.ready_event.set()
            if self.verbose:
                print(f"[FTSensorController] Disconnected from FT sensor.")