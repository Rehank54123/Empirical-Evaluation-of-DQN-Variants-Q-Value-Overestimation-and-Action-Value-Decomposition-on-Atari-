"""
Standard DeepMind Atari Preprocessing Wrappers for Gymnasium.
Conforms to the preprocessing specifications introduced in Mnih et al. (Nature 2015).
Compatible with gymnasium and ale-py.
"""

import os
import shutil
import tempfile
from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box

# Fix for Atari/Stella ROM loading on Windows when paths contain non-ASCII / Unicode characters
try:
    import ale_py
    from ale_py import roms
    gym.register_envs(ale_py)

    _rom_cache_dir = os.path.join(tempfile.gettempdir(), "ale_roms_cache")
    os.makedirs(_rom_cache_dir, exist_ok=True)
    _orig_get_rom_path = roms.get_rom_path

    def _safe_get_rom_path(game: str):
        target = os.path.join(_rom_cache_dir, f"{game}.bin")
        if not os.path.exists(target):
            src = _orig_get_rom_path(game)
            shutil.copyfile(src, target)
        return target

    roms.get_rom_path = _safe_get_rom_path
except Exception:
    pass



class NoopResetEnv(gym.Wrapper):
    """
    Sample initial NOOP actions on reset to introduce environmental stochasticity.
    Prevents the agent from overfitting to a single deterministic initial trajectory.
    """
    def __init__(self, env, noop_max: int = 30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        info = {}
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Press FIRE on reset for environments that require FIRE to initiate gameplay
    (e.g., Breakout, Pong).
    """
    def __init__(self, env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        assert "FIRE" in meanings
        self.fire_action = meanings.index("FIRE")

    def reset(self, **kwargs):
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(self.fire_action)
        if terminated or truncated:
            self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(0)  # NOOP
        if terminated or truncated:
            self.env.reset(**kwargs)
        return obs, info


class EpisodicLifeEnv(gym.Wrapper):
    """
    Treat loss of life as end of episode for training purposes.
    Helps value estimation recognize that losing a life carries severe future consequence,
    without terminating the underlying environment until all lives are exhausted.
    """
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # Life lost: mark as terminated for credit assignment during training
            terminated = True
        self.lives = lives
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # Resuming play within the same game session after a life loss
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame and max-pool over the last 2 frames.
    Max-pooling over the last 2 frames eliminates phosphor flickering artifacts
    inherent to the Atari 2600 hardware.
    """
    def __init__(self, env, skip: int = 4):
        super().__init__(env)
        self._obs_buffer = np.zeros((2, *env.observation_space.shape), dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        for i in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            total_reward += reward
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            terminated = term
            truncated = trunc
            if terminated or truncated:
                break
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip reward to sign(r) or in {-1, 0, +1} to normalize gradients across diverse games.
    """
    def __init__(self, env):
        super().__init__(env)

    def reward(self, reward):
        return np.sign(reward)


class WarpFrame(gym.ObservationWrapper):
    """
    Converts 210x160 RGB screen observation to 84x84 Grayscale.
    Uses cv2 if available, otherwise falls back to PIL / numpy.
    """
    def __init__(self, env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        self.observation_space = Box(
            low=0, high=255, shape=(self.height, self.width), dtype=np.uint8
        )

    def observation(self, frame):
        # Convert RGB to grayscale (standard ITU-R 601-2 luma transform)
        # Y = 0.299 R + 0.587 G + 0.114 B
        if frame.ndim == 3 and frame.shape[2] == 3:
            gray = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        else:
            gray = frame

        # Resize to 84x84
        try:
            import cv2
            resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
        except ImportError:
            from PIL import Image
            img = Image.fromarray(gray)
            resized = np.array(img.resize((self.width, self.height), Image.BILINEAR), dtype=np.uint8)
        return resized


class FrameStack(gym.Wrapper):
    """
    Stack `k` consecutive frames along the channel axis.
    Provides the network with temporal context (ball direction, velocity, object motion).
    Output shape: (k, 84, 84)
    """
    def __init__(self, env, k: int = 4):
        super().__init__(env)
        self.k = k
        self.frames = deque(maxlen=k)
        orig_shape = env.observation_space.shape
        self.observation_space = Box(
            low=0,
            high=255,
            shape=(k, *orig_shape),
            dtype=env.observation_space.dtype,
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.k):
            self.frames.append(obs)
        return self._get_ob(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_ob(), reward, terminated, truncated, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return np.array(self.frames)


def make_atari_env(
    env_id: str,
    is_eval: bool = False,
    noop_max: int = 30,
    frame_skip: int = 4,
    stack_frames: int = 4,
    clip_rewards: bool = True
):
    """
    Factory function producing standard DeepMind-compliant Atari environment wrappers.
    Args:
        env_id: Gymnasium environment ID (e.g. 'ALE/Pong-v5' or 'PongNoFrameskip-v4')
        is_eval: If True, disables EpisodicLifeEnv and ClipRewardEnv to record true episode returns.
    """
    env = gym.make(env_id)
    if noop_max > 0:
        env = NoopResetEnv(env, noop_max=noop_max)
    if frame_skip > 0:
        env = MaxAndSkipEnv(env, skip=frame_skip)
    
    if not is_eval:
        env = EpisodicLifeEnv(env)
    
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
        
    env = WarpFrame(env, width=84, height=84)
    
    if not is_eval and clip_rewards:
        env = ClipRewardEnv(env)
        
    if stack_frames > 0:
        env = FrameStack(env, k=stack_frames)
        
    return env

