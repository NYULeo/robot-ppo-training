

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
from typing import Dict, Tuple, Any, Optional, List
import xml.etree.ElementTree as ET
from scipy.spatial.transform import Rotation as R
import torch
import torch.distributions
from DNN import Actor
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle
import time
import json
from pathlib import Path
import imageio.v3 as iio
import mediapy as media

class H780MuJoCoEnv(gym.Env):
    """
    MuJoCo-based environment for the H780 humanoid robot with animation capabilities.
    
    State Space (46 dimensions):
    - Joint positions (23)
    - Joint velocities (23) 
    
    Action Space (23 dimensions):
    - Joint torques for all 23 actuated joints
    """
    
    def __init__(self, 
                 urdf_path: str = "H780bv2.SLDASM/H780bv2.SLDASM.xml",
                 control_frequency: int = 100,
                 physics_frequency: int = 1000,
                 max_episode_steps: int = 1000,
                 render_mode: Optional[str] = None,
                 enable_animation: bool = False,
                 animation_save_path: str = "animations"):
        
        super().__init__()
        
        # Environment parameters
        self.control_frequency = control_frequency
        self.physics_frequency = physics_frequency
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self.dt = 1.0 / control_frequency
        
        # Animation parameters
        self.enable_animation = enable_animation
        self.animation_save_path = Path(animation_save_path)
        self.animation_save_path.mkdir(exist_ok=True)
        
        # Animation data storage
        self.animation_data = {
            'observations': [],
            'actions': [],
            'rewards': [],
            'base_positions': [],
            'base_orientations': [],
            'joint_positions': [],
            'joint_velocities': [],
            'timestamps': []
        }
        
        # URDF path
        self.urdf_path = urdf_path
        
        # Joint information
        self.joint_names = [
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_pitch_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint', 
            'right_knee_pitch_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'waist_yaw_joint', 'waist_roll_joint', 'waist_pitch_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_pitch_joint', 'right_shoulder_pitch_joint', 'right_shoulder_roll_joint',
            'right_shoulder_yaw_joint', 'right_elbow_pitch_joint'
        ]
        

        
        
        # Joint limits from URDF analysis
        self.joint_limits = {
            'lower': np.array([
                -2.5307, -0.5236, -2.7576, -0.0872, -0.5236, -0.5236,  # left leg
                -2.5307, -0.5236, -2.7576, -0.0872, -0.5236, -0.5236,  # right leg
                -2.7576, -0.5236, -0.5236,  # waist
                -2.5307, -0.5236, -2.7576, -0.0872,  # left arm
                -2.5307, -0.5236, -2.7576, -0.0872   # right arm
            ]),
            'upper': np.array([
                2.8798, 2.9671, 2.7576, 2.5307, 0.5236, 0.5236,  # left leg
                2.8798, 2.9671, 2.7576, 2.5307, 0.5236, 0.5236,  # right leg
                2.7576, 0.5236, 0.5236,  # waist
                2.8798, 0.5236, 2.7576, 0.0872,  # left arm
                2.8798, 0.5236, 2.7576, 0.0872   # right arm
            ]),
            'effort': np.array([
                48, 14, 14, 48, 14, 14,  # left leg
                48, 14, 14, 48, 14, 14,  # right leg
                17, 14, 14,  # waist
                5.5, 14, 5.5, 5.5,  # left arm
                5.5, 14, 5.5, 5.5   # right arm
            ]),
            'velocity': np.array([
                12.566, 27.227, 27.227, 12.566, 27.227, 27.227,  # left leg
                12.566, 27.227, 27.227, 12.566, 27.227, 27.227,  # right leg
                28.798, 27.227, 27.227,  # waist
                10.472, 27.227, 10.472, 10.472,  # left arm
                10.472, 27.227, 10.472, 10.472   # right arm
            ])
        }
        
        # Initialize MuJoCo
        self._initialize_mujoco()
        
        # Define spaces
        self._define_spaces()
        
    def _load_robot_modelx(self):
        """Load and parse the H780 robot URDF model."""
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF file not found: {self.urdf_path}")
        
        # Parse URDF to extract joint information
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        self.joint_info = {}
        for joint in root.findall('joint'):
            joint_name = joint.get('name')
            joint_type = joint.get('type')
            
            if joint_type == 'revolute':
                limits = joint.find('limit')
                if limits is not None:
                    self.joint_info[joint_name] = {
                        'type': joint_type,
                        'lower': float(limits.get('lower')),
                        'upper': float(limits.get('upper')),
                        'effort': float(limits.get('effort')),
                        'velocity': float(limits.get('velocity'))
                    }
    
    def _initialize_mujoco(self):
        """Initialize MuJoCo model and data."""
        try:
            # Load the model
            self.model = mujoco.MjModel.from_xml_path(self.urdf_path)
            self.data = mujoco.MjData(self.model)
            
            """
            # Create MuJoCo context for rendering
            if self.render_mode == "human":
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            """
            # Set simulation parameters
            self.model.opt.timestep = 1.0 / self.physics_frequency
            self.model.opt.iterations = 20
            self.model.opt.solver = mujoco.mjtSolver.mjSOL_CG
            
            # Set gravity
            self.model.opt.gravity[2] = -9.81
            
            print(f"MuJoCo model loaded successfully!")
            print(f"Number of DOFs: {self.model.nv}")
            print(f"Number of bodies: {self.model.nbody}")
            print(f"Number of joints: {self.model.njnt}")
            
        except Exception as e:
            print(f"Error loading MuJoCo model: {e}")
            raise RuntimeError("Failed to initialize MuJoCo physics engine")
    
    def _define_spaces(self):
        """Define action and observation spaces."""
        # Action space: normalized actions in [-1, 1] range
        # These will be scaled to actual joint limits during execution
        action_low = np.ones(23) * -1.0
        action_high = np.ones(23) * 1.0
        
        self.action_space = spaces.Box(
            low=action_low,
            high=action_high,
            dtype=np.float32
        )
        
        # Observation space: 59 dimensions
        # Joint positions (23) + joint velocities (23) + base states (13)
        obs_low = np.concatenate([
            self.joint_limits['lower'],  # joint positions
            -self.joint_limits['velocity'],  # joint velocities
        ])
        
        obs_high = np.concatenate([
            self.joint_limits['upper'],  # joint positions
            self.joint_limits['velocity'],  # joint velocities
        ])
        
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32
        )
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)
        
        # Check if custom state is provided in options
        custom_state = None
        if options and 'custom_state' in options:
            custom_state = options['custom_state']
        
        # Reset MuJoCo data
        mujoco.mj_resetData(self.model, self.data)
        
        if custom_state is not None:
            self._set_state_mujoco(custom_state)
        else:
            # Set initial joint positions (standing pose)
            initial_qpos = np.zeros(self.model.nq)
            initial_qvel = np.zeros(self.model.nv)
            
            # Set a more stable standing pose
            # Slightly bend knees and hips for stability
            initial_qpos[3] = -0.1   # left_knee_pitch (slight bend)
            initial_qpos[9] = -0.1   # right_knee_pitch (slight bend)
            initial_qpos[0] = 0.1    # left_hip_pitch (slight forward)
            initial_qpos[6] = 0.1    # right_hip_pitch (slight forward)
            
            # Set initial state
            self.data.qpos[:] = initial_qpos
            self.data.qvel[:] = initial_qvel
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
        
        # Reset episode tracking
        self.episode_steps = 0
        self.episode_reward = 0.0
        
        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def set_state(self, state: np.ndarray):
        """
        Set the environment to an arbitrary state.
        
        Args:
            state: 46-dimensional state vector containing:
                - joint_positions (23)
                - joint_velocities (23) 
        """
        if len(state) != 46:
            raise ValueError(f"State must be 46-dimensional, got {len(state)}")
        
        self._set_state_mujoco(state)
    
    def _set_state_mujoco(self, state: np.ndarray):
        """Set state for MuJoCo physics."""
        # Extract components from state vector
        joint_pos = state[:23]
        joint_vel = state[23:46]
       
    
        
        # Clamp joint positions to limits
        joint_pos = np.clip(joint_pos, self.joint_limits['lower'], self.joint_limits['upper'])
        
        # Set MuJoCo state
        self.data.qpos[:23] = joint_pos
        self.data.qvel[:23] = joint_vel
        
        
        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: Joint torques for all 23 actuated joints
            
        Returns:
            observation: Current state observation
            reward: Reward for this step
            terminated: Whether episode has terminated
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Validate action
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} not in action space {self.action_space}")
        
        # Scale action to joint torques
        scaled_action = action * self.joint_limits['effort']
        
        # Apply action (joint torques)
        self.data.ctrl[:] = scaled_action
        # Step simulation
        mujoco.mj_step(self.model, self.data)
        
        # Get observation
        observation = self._get_observation()
        
        # Calculate reward
        reward = self._compute_reward(observation, action)
        
        # Check termination conditions
        terminated = self._check_termination()
        
        # Check truncation
        self.episode_steps += 1
        truncated = self.episode_steps >= self.max_episode_steps
        
        # Update episode reward
        self.episode_reward += reward
        
        # Get info
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _check_termination(self) -> bool:
        """Check if episode should terminate early."""
        # Get base position and orientation
        base_pos = self.data.xpos[0].copy()
        base_quat = self.data.xquat[0].copy()
        
        # Check if robot has fallen (base too low)
        if base_pos[2] < 0.5:  # Base height threshold
            return True
        
        # Check if robot has fallen over (excessive roll/pitch)
        # Convert quaternion to euler angles for easier interpretation
        
        rot = R.from_quat([base_quat[1], base_quat[2], base_quat[3], base_quat[0]])  # MuJoCo uses [w,x,y,z]
        euler = rot.as_euler('xyz')
        
        # Check roll and pitch angles (excessive tilt)
        if abs(euler[0]) > 0.8 or abs(euler[1]) > 0.8:  # ~45 degrees
            return True
        
        # Check if robot has moved too far from origin
        if np.linalg.norm(base_pos[:2]) > 10.0:  # Too far in x,y
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Extract observation from current state."""
        # Use MuJoCo state
        # Joint positions and velocities
        joint_pos = self.data.qpos[:23].copy()  # First 23 are actuated joints
        joint_vel = self.data.qvel[:23].copy()
        
       
        # Combine into observation vector
        observation = np.concatenate([
            joint_pos,
            joint_vel
        ])
        
        return observation.astype(np.float32)
    

    
    def _compute_reward(self, obs: np.ndarray, action: np.ndarray):
       
       target_cadence_rad_s = 2.0
       w_speed = 1.0
       w_phase = 0.1
       w_posture = 0.2
       w_alive = 0.0  # Remove constant alive bonus to prevent reward explosion
       w_effort = 1e-3  # Increase effort penalty
       assert len(obs) == 46 and len(action) == 23
       qpos = obs[:23]
       qvel = obs[23:46]

       # 1) "speed" surrogate (fixed-base proxy): hip-pitch angular speed
       vL = abs(qvel[0])    # left_hip_pitch
       vR = abs(qvel[6])    # right_hip_pitch
       cadence = 0.5 * (vL + vR)
       sigma = max(0.25 * target_cadence_rad_s, 0.2)
       r_speed = np.exp(-((cadence - target_cadence_rad_s) ** 2) / (2 * sigma ** 2))

       # 2) left–right anti-phase on legs (encourage Δθ ≈ π)
       r_pairs = []
       for i in range(6):                       # pairs: 0..5 vs 6..11
           d = qpos[i] - qpos[6 + i]
           r_pairs.append(0.5 * (1.0 - np.cos(d)))   # (1 - cosΔ)/2 ∈ [0,1]
       r_phase = float(np.mean(r_pairs))

       # 3) upright-ish posture via small waist roll/pitch
       waist_roll  = qpos[13]
       waist_pitch = qpos[14]
       r_posture = np.exp(-(abs(waist_roll) + abs(waist_pitch)))  # smooth in (0,1]

       # 4) alive bonus (small constant) - REMOVED to prevent explosion
       r_alive = 0.0

       # 5) effort cost (no prev action term)
       cost_effort = float(np.sum(action ** 2))

       # 6) Add height bonus to encourage staying upright
       base_height = self.data.xpos[0][2]  # Base height
       r_height = np.clip(base_height / 2.0, 0.0, 1.0)  # Normalize height reward
       w_height = 0.1

       reward = (w_speed * r_speed
              + w_phase * r_phase
              + w_posture * r_posture
              + w_alive * r_alive
              + w_height * r_height
              - w_effort * cost_effort)

       # Clip reward to prevent extreme values
       reward = np.clip(reward, -10.0, 10.0)
       
       return float(reward)



    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information about the environment."""
        return {
            'episode_steps': self.episode_steps,
            'episode_reward': self.episode_reward,
            'base_position': self.data.xpos[0].copy(),
            'base_orientation': self.data.xquat[0].copy(),
            'joint_positions': self.data.qpos[:23].copy(),
            'joint_velocities': self.data.qvel[:23].copy(),
        }
    
    def render(self):
        """Render the environment."""
        if self.render_mode == "human" and hasattr(self, 'viewer'):
            self.viewer.sync()
    
    def close(self):
        """Close the environment."""
        if hasattr(self, 'viewer'):
            self.viewer.close()


if __name__ == "__main__":
    # Create environment
    env = H780MuJoCoEnv(render_mode="human")
    
    print("Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")
    
    # Test reset
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial base position: {info['base_position']}")
    
    # Test setting arbitrary state
    print("\n=== Testing Arbitrary State Setting ===")
    
    # Create a custom state (sitting pose)
    custom_state = np.zeros(46)
    
   
    
    # Set the custom state
    env.set_state(custom_state)
    
    # Get observation after setting custom state
    obs_after = env._get_observation()
    info_after = env._get_info()
    
    print(f"Custom state base position: {info_after['base_position']}")
    print(f"Custom state joint positions: {info_after['joint_positions'][:6]}")  # Show first 6 joints
    
    # Test a few steps with random actions
    state_dim = 46
    action_dim = 23
    actor = Actor(state_dim, action_dim)

# Load weights
    obs = obs_after
    actor.load_state_dict(torch.load("ppo_actor.pth"))
    actor.eval()  # evaluation mode
    print("\n=== Testing Random Actions ===")
    
   
    # NEW: create a renderer and a list to hold frames
    renderer = mujoco.Renderer(env.model)  # tweak size if you like
    frames = []

    print("\n=== Testing Actor Policy & Recording ===")
    for step in range(1000):
    # your actor
       obs_t = torch.from_numpy(obs).float()
       mean, std = actor(obs_t)
       dist = torch.distributions.MultivariateNormal(mean.squeeze(0),
                                                  covariance_matrix=torch.diag(std**2))
       action = dist.mean.cpu().detach().numpy()  # deterministic mean action

    # step env (your API)
       obs, reward, terminated, truncated, info = env.step(action)

    # NEW: render the current MuJoCo state into an RGB frame
    # Ensure forward kinematics is up-to-date
       mujoco.mj_forward(env.model, env.data)
      
    # If you have a camera named in your MJCF, pass camera="your_camera_name" or an int
    # e.g., camera="track" or camera=0. If not, omit camera=... to use default.
       renderer.update_scene(env.data)               # or: renderer.update_scene(env.data, camera="track")
       frame = renderer.render()                     # HxWx3 uint8
       frames.append(frame)

       if step % 10 == 0:
          print(f"Step {step}: Reward = {reward:.3f}, Base height = {info['base_position'][2]:.3f}")

# --- after the loop, write the animation ---
# --- after the loop, write the animation ---
# MP4 (needs ffmpeg available); if not available, write GIF instead.
#iio.imwrite("rollout.mp4", frames, fps=30, codec="libx264")   # try MP4 first
# Also write a GIF (easier to preview, larger file):
#iio.imwrite("rollout.gif", frames, fps=15)
    media.write_video("demo.mp4", frames, fps=100)












