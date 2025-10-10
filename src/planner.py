"""
Bipedal robot footstep planner using stable-baselines3 with CNN feature extractor.

Implements a custom CNN+MLP architecture for processing terrain observations and vector data.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

try:
    from .environment import BipedEnvironment
except:
    from environment import BipedEnvironment


class CustomCnnExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Dict observations with terrain image and vector data.
    
    Architecture:
    - CNN processes (1, 200, 200) terrain patch
    - CNN features are flattened and concatenated with vector data
    - MLP processes combined features
    """
    
    def __init__(self, observation_space, features_dim: int = 128):
        # Extract individual spaces with runtime check
        if not hasattr(observation_space, 'spaces'):
            raise ValueError("Expected Dict observation space")
        
        vector_space = observation_space.spaces["vector"]
        vector_dim = vector_space.shape[0]
        
        # Calculate CNN output dimensions
        # Input: (1, 200, 200)
        # After conv1: (16, 50, 50)   # 200/4 = 50
        # After conv2: (32, 25, 25)   # 50/2 = 25  
        # After conv3: (64, 12, 12)   # 25/2 = 12.5 -> 12
        # After adaptive pool: (64, 2, 2)  # Force to 2x2
        # Flatten: 64 * 2 * 2 = 256
        
        super().__init__(observation_space, features_dim)
        
        # CNN for terrain processing - aggressive reduction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=8, stride=4, padding=2),  # 200 -> 50
            nn.ReLU(),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2),  # 50 -> 25
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 25 -> 12
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((2, 2)),  # Force to 2x2 = 256 dims
            nn.Flatten()
        )
        
        # Calculate CNN output size
        with torch.no_grad():
            sample_terrain = torch.zeros(1, 1, 200, 200)
            cnn_output = self.cnn(sample_terrain)
            cnn_output_dim = cnn_output.shape[1]
        
        # MLP for combined features - smaller due to reduced CNN output
        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_dim + vector_dim, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
    
    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            observations: Dict with 'terrain' and 'vector' keys
            
        Returns:
            Combined features tensor
        """
        terrain = observations["terrain"]  # Shape: (batch, 1, 200, 200)
        vector = observations["vector"]    # Shape: (batch, 5) or (5,)
        
        # Ensure vector has batch dimension
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)  # Shape: (1, 5)
        
        # Process terrain through CNN
        terrain_features = self.cnn(terrain)
        
        # Concatenate with vector data
        combined = torch.cat([terrain_features, vector], dim=1)
        
        # Process through MLP
        features = self.mlp(combined)
        
        return features


def create_planner(env: Optional[Any] = None, 
                   algorithm: str = "PPO",
                   learning_rate: float = 3e-4,
                   n_steps: int = 512,
                   batch_size: int = 512,
                   verbose: int = 1) -> Any:
    """
    Create a stable-baselines3 planner with custom CNN extractor.
    
    Args:
        env: Environment instance (creates default if None)
        algorithm: RL algorithm to use ("PPO" recommended)
        learning_rate: Learning rate for training
        n_steps: Number of steps per update
        batch_size: Batch size for training
        verbose: Verbosity level
        
    Returns:
        Trained model instance
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    
    # Create environment if not provided
    if env is None:
        env = BipedEnvironment()
    
    # Wrap in DummyVecEnv if not already a VecEnv
    if not hasattr(env, 'num_envs'):
        env = DummyVecEnv([lambda: env])
    
    # Create policy kwargs with custom feature extractor
    policy_kwargs = dict(
        features_extractor_class=CustomCnnExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )
    
    # Create model based on algorithm
    if algorithm.upper() == "PPO":
        # Auto-detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        model = PPO(
            MultiInputActorCriticPolicy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=None,  # Disable tensorboard for now
            device=device
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}. Use 'PPO'.")
    
    return model


def test_planner():
    """Test the planner implementation with basic functionality."""
    print("Creating environment (no rendering)...")
    env = BipedEnvironment(render_mode=None)  # Avoid pygame initialization issues
    
    print("Creating planner...")
    model = create_planner(env, verbose=1)
    
    print("Testing observation processing...")
    obs, _ = env.reset()
    print(f"Observation keys: {obs.keys()}")
    print(f"Terrain shape: {obs['terrain'].shape}")
    print(f"Vector shape: {obs['vector'].shape}")
    
    print("Testing model prediction...")
    action, _ = model.predict(obs, deterministic=True)
    print(f"✓ Prediction successful! Action: {action}")
    
    print("Testing single step...")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful! Reward: {reward:.2f}")
    
    print("Planner test completed successfully!")


if __name__ == "__main__":
    test_planner()