# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a bipedal robot footstep planning project that implements approaches for generating footstep sequences on 3D terrain. The project contains a main implementation area and a complete reference implementation using reinforcement learning.

## Architecture

### Main Project Structure

- `main.py` - Entry point (currently placeholder)
- `src/` - Main implementation directory (empty, ready for development)
- `pyproject.toml` - Modern Python project configuration with uv package manager

### Reference Implementation (`reference/`)

Complete RL-based footstep planning system with 15 registered environment variants:

- Base environments: `footsteps-planning-{right,left,any}-v0`
- Multi-goal variants: `footsteps-planning-{right,left,any}-multigoal-v0`
- Obstacle variants: `footsteps-planning-{right,left,any}-obstacle-multigoal-v0`

**Core Components:**

- `FootstepsPlanningEnv` - Base environment with 8D observation space and 3D action space
- `Simulator` - 2D physics simulation with realistic foot geometry and collision detection
- Multiple training algorithms supported (TD3, CrossQ, DDPG, SAC, etc.)

## Development Commands

### Package Management

```bash
# Install dependencies
uv sync

# Run the main project
python main.py

# Run reference implementation demo
cd reference && python demo.py
```

### Training Commands

```bash
# Using Stable-Baselines-Jax (primary approach)
cd reference
python train_sbx.py --algo crossq --env footsteps-planning-any-v0 --conf hyperparams/crossq.yml

# Using RL-Zoo3 (alternative)
python -m rl_zoo3.train --algo td3 --env footsteps-planning-right-v0 --gym-packages gym_footsteps_planning --conf hyperparams/td3.yml
```

### Evaluation and Testing

```bash
# Enjoy trained models
python enjoy_sbx.py --algo crossq --env footsteps-planning-any-v0 --load-best

# Code quality checks
make commit-checks  # Runs ruff, black, mypy
```

## Assignment Integration

The project is set up to implement a custom footstep planning solution with these specific constraints:

- Foot size: 3x5
- Maximum step length: 40
- Inter-foot distance: 2-10
- Observation range: 200x200 region ahead
- Foot plane normal angle: ±20 degrees from gravity

Model input:

- 3D terrain map input with start/end projected coordinates

Model output:

- Sequence of bipedal footsteps

The assignment also requires:

- Visualization of foot center position coordinates and angle between foot plane normal direction and gravity direction
- Trajectory length
- Turning angle curve
- Curve of angle between foot plane normal direction and gravity direction
- Experimental data analysis and safety analysis

## Model Architecture

**Reference Implementation Uses Pre-built Networks:**

- The reference uses StableBaselines3's built-in `MlpPolicy`
- Network architecture: 2-layer MLP with [384, 256] units as specified in hyperparams/crossq.yml
- No custom model definition needed - algorithms handle network creation internally

**For Custom Implementation:**

- You can define custom PyTorch networks by extending `BasePolicy` from StableBaselines3
- Networks should accept your custom observation space (terrain + foot state)
- Output action space: [dx, dy, dtheta] for footstep displacement

## Development Workflow

### 1. Environment Development (`src/`)

Create custom 3D terrain environment mirroring FootstepsPlanningEnv structure:

- Implement `step()`, `reset()`, `render()` methods following gymnasium.Env interface
- Add 3D terrain processing and height-based foot placement validation
- Enforce assignment-specific constraints (foot size 3x5, step length 40, ±20° angle)

### 2. Custom Model Definition (`src/models/`)

Define PyTorch networks for RL algorithms:

- Custom feature extractor for terrain observations
- Policy network with appropriate input/output dimensions
- Value network for critic in actor-critic methods

### 3. Terrain Generation (`src/terrain/`)

Implement flexible terrain generation:

- Random height maps with varying difficulty
- Smooth and rough terrain types
- Configurable terrain parameters for testing

### 4. Logging and Visualization (`src/utils/`)

Enhanced logging and visualization systems:

- **SwanLab integration**: Track experiments with custom metrics
- **File logging**: Save training logs to `.txt` files alongside terminal output
- **Assignment visualizations**:
  - Footstep sequence visualization with foot center positions and angles
  - Trajectory length plotting
  - Turning angle curves
  - Foot plane normal angle curves vs gravity
  - Safety analysis visualizations

### 5. Training Infrastructure

Use StableBaselines3 algorithms with custom configuration:

- **Recommended algorithms**: TD3 (from reference) or CrossQ (SBX)
- **Training pipeline**: Automated hyperparameter management
- **Model saving**: Checkpointing with SwanLab integration
- **Evaluation**: Comprehensive metrics and assignment requirement validation

## Implementation Commands

```bash
# Train with custom environment and logging
cd src
python train.py --env CustomFootstepEnv --algo td3 --config configs/custom_config.yml

# Generate flexible terrain
python terrain/generate_terrain.py --type rough --size 100x100 --output terrain_data/

# Run evaluation with assignment visualizations
python evaluate.py --model_path models/custom_model.zip --terrain terrain_data/test_terrain.npz
```

## Key Implementation Notes

1. **Environment Mirroring**: Your custom env should follow the same interface as FootstepsPlanningEnv but with 3D terrain processing and your specific constraints
2. **Model Flexibility**: While the reference uses built-in MlpPolicy, you can define custom networks for terrain feature extraction
3. **Terrain-Centric**: Unlike the reference's "obstacle" approach, your environment uses terrain heights as the primary challenge
4. **Assignment Compliance**: Ensure all required outputs are generated during evaluation and saved for analysis

The reference implementation provides excellent patterns for RL training, environment design, and production deployment that can be adapted to your 3D terrain requirements.
