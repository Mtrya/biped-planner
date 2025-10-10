"""
Main entry point for biped planner training and demonstration.
"""

import argparse
import os

from src.environment import BipedEnvironment
from src.planner import create_planner


def train_model(timesteps: int = 10000, save_path: str = "models/biped_planner.zip"):
    """Train the biped planner model."""
    print("Training Biped Planner")
    print("=" * 40)
    
    # Create environment (no rendering for training)
    print("Creating environment...")
    env = BipedEnvironment()
    
    # Create planner
    print("Creating planner...")
    model = create_planner(env, verbose=2)
    
    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Train the model
    print(f"Training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps)
    
    # Save the model
    print(f"Saving model to {save_path}...")
    model.save(save_path)
    
    print("Training completed!")
    env.close()


def demo_model(model_path: str = "models/biped_planner.zip"):
    """Run demonstration with trained model."""
    print("Biped Planner Demonstration")
    print("=" * 40)
    
    # Create environment with visualization
    print("Creating environment...")
    env = BipedEnvironment(render_mode="human")
    
    # Create planner and load trained model
    print("Loading planner...")
    model = create_planner(env, verbose=0)
    
    if os.path.exists(model_path):
        print(f"Loading trained model from {model_path}...")
        model.set_parameters(model_path)
    else:
        print(f"No trained model found at {model_path}, using untrained model...")
    
    # Reset environment
    print("Resetting environment...")
    obs, _ = env.reset()
    
    # Run demonstration
    print("Running planner for 200 steps...")
    for step in range(200):
        # Get action from planner
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Render
        env.render()
        
        print(f"Step {step + 1}: reward={reward:.2f}, terminated={terminated}")
        
        if terminated:
            print("Goal reached!")
            break
    
    print("Demonstration completed!")
    env.close()


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description="Biped Planner - Train or Demo")
    parser.add_argument(
        "mode", 
        choices=["train", "demo"], 
        help="Mode: train the model or run demonstration"
    )
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=10000,
        help="Number of training timesteps (default: 10000)"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="models/biped_planner.zip",
        help="Path to save/load model (default: models/biped_planner.zip)"
    )
    
    args = parser.parse_args()
    
    if args.mode == "train":
        train_model(args.timesteps, args.model_path)
    elif args.mode == "demo":
        demo_model(args.model_path)


if __name__ == "__main__":
    main()
