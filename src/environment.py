"""
Bipedal robot footstep planning environment.

Implements continuous state/action space with terrain integration and constraint checking.
"""

import gymnasium
import numpy as np
import math
from typing import Tuple, Optional, Dict
from gymnasium import spaces
import pygame
import os

try:
    from .terrain import get_terrain, get_height
except:
    from terrain import get_terrain, get_height


class BipedEnvironment(gymnasium.Env):
    """
    Simple bipedal footstep planning environment with continuous state/action space.

    State: (x, y, theta, foot_side) where x in [0,3000], y in [0,1000], theta in [0,2pi), foot_side in {0,1}
    Action: (dx, dy, dtheta) with physical constraints
    Observation: 200x200 terrain patch + goal/start in local coordinates
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = None):
        # Terrain integration
        self.terrain = get_terrain()

        self.max_step_length = 40.0
        self.min_foot_distance = 2.0
        self.max_foot_distance = 10.0
        self.max_turn_angle = math.radians(75)

        self.obs_range = 200  # 200x200 observation window
        self.terrain_width = 1000  # y in [0,1000]
        self.terrain_height = 3000  # x in [0,3000]

        # Set up scale factor (for rendering)
        self.scale_factor = 0.2

        # Generate terrain background image
        self.terrain_image_path = "terrain_background.png"
        self._generate_terrain_image()

        self.action_space = spaces.Box(
            low=np.array([-self.max_foot_distance, -self.max_foot_distance, -self.max_turn_angle]),
            high=np.array([self.max_foot_distance, self.max_foot_distance, self.max_turn_angle]),
            dtype=np.float64
        )

        # Observation space: 200x200 terrain height + start/goal info
        # Terrain patch flattened + start_x, start_y, goal_x, goal_y, current_foot_side
        obs_size = self.obs_range * self.obs_range + 5
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float64
        )

        # State variables
        self.position = np.array([0.0, 0.0])  # (x, y)
        self.orientation = 0.0  # theta
        self.foot_side = 0  # 0=left, 1=right
        self.start_pos = np.array([0.0, 500.0])  # Default start (x=0, y=500)
        self.goal_pos = np.array([3000.0, 500.0])  # Default goal (x=3000, y=500)
        self.trajectory = []  # Store footstep history

        # Rendering
        self.render_mode = render_mode

        # Initialize
        self.reset()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        if options:
            self.start_pos = np.array(options.get('start_pos', self.start_pos))
            self.goal_pos = np.array(options.get('goal_pos', self.goal_pos))

        # Reset state to start position
        self.position = self.start_pos.copy()
        self.orientation = math.pi / 2  # Face upward initially
        self.foot_side = 0  # Start with left foot
        self.trajectory = [self.position.copy()]

        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step with given action."""
        dx, dy, dtheta = action

        # Calculate new position and orientation
        new_orientation = self.orientation + dtheta
        new_position = self.position + np.array([dx, dy])

        # Validate constraints
        if not self._validate_step(dx, dy, dtheta):
            # Invalid step - penalty and no movement
            reward = -10.0
            terminated = False
            truncated = False
            info = {'reason': 'constraint_violation', 'constraint_details': self._get_constraint_details(dx, dy, dtheta)}
            return self._get_observation(), reward, terminated, truncated, info

        # Update state
        self.position = new_position
        self.orientation = new_orientation % (2 * math.pi)
        self.foot_side = 1 - self.foot_side  # Switch foot
        self.trajectory.append(self.position.copy())

        # Check if reached goal
        distance_to_goal = np.linalg.norm(self.position - self.goal_pos)
        reached_goal = distance_to_goal < 2.0

        # Calculate reward
        if reached_goal:
            reward = 100.0
            terminated = True
        else:
            # Progress reward: progress toward goal minus step penalty
            prev_distance = np.linalg.norm(self.trajectory[-2] - self.goal_pos) if len(self.trajectory) > 1 else distance_to_goal
            progress = prev_distance - distance_to_goal
            reward = progress - 0.1  # Small step cost
            terminated = False

        truncated = False
        info = {
            'position': self.position.copy(),
            'orientation': self.orientation,
            'foot_side': self.foot_side,
            'distance_to_goal': distance_to_goal,
            'trajectory_length': len(self.trajectory)
        }

        return self._get_observation(), reward, terminated, truncated, info

    def _validate_step(self, dx: float, dy: float, dtheta: float) -> bool:
        """Validate step against all constraints."""
        # 1. Turn angle constraint
        if abs(dtheta) > self.max_turn_angle:
            return False

        # 2. Step length constraint (3D distance)
        new_position = self.position + np.array([dx, dy])
        current_height = get_height(self.position[0], self.position[1])
        new_height = get_height(new_position[0], new_position[1])
        step_3d = math.sqrt(dx**2 + dy**2 + (new_height - current_height)**2)

        if step_3d > self.max_step_length:
            return False

        # 3. Foot distance constraint (check previous foot position)
        if len(self.trajectory) > 1:  # Only check after at least 2 steps
            prev_foot_pos = self.trajectory[-2]  # Previous foot position
            foot_distance = np.linalg.norm(new_position - prev_foot_pos)

            if foot_distance < self.min_foot_distance or foot_distance > self.max_foot_distance:
                return False

        # 4. Boundary constraints
        if (new_position[0] < 0 or new_position[0] >= self.terrain_height or
            new_position[1] < 0 or new_position[1] >= self.terrain_width):
            return False

        return True

    def _get_constraint_details(self, dx: float, dy: float, dtheta: float) -> Dict[str, float]:
        """Get detailed constraint violation information for debugging."""
        details = {}

        # Turn angle
        details['turn_angle'] = math.degrees(abs(dtheta))
        details['turn_angle_limit'] = math.degrees(self.max_turn_angle)

        # Step length
        new_position = self.position + np.array([dx, dy])
        current_height = get_height(self.position[0], self.position[1])
        new_height = get_height(new_position[0], new_position[1])
        step_3d = math.sqrt(dx**2 + dy**2 + (new_height - current_height)**2)
        details['step_length'] = step_3d
        details['step_length_limit'] = self.max_step_length

        # Foot distance
        if len(self.trajectory) > 0:
            prev_foot_pos = self.trajectory[-2] if len(self.trajectory) > 1 else self.start_pos
            foot_distance = np.linalg.norm(new_position - prev_foot_pos)
            details['foot_distance'] = foot_distance
            details['foot_distance_min'] = self.min_foot_distance
            details['foot_distance_max'] = self.max_foot_distance

        
        return details

    def _get_observation(self) -> np.ndarray:
        """Generate current observation: terrain patch + start/goal info."""
        # Extract 200x200 terrain patch centered at current position
        terrain_patch = self._get_terrain_patch()

        # Transform start and goal to local coordinates
        start_local = self._global_to_local(self.start_pos)
        goal_local = self._global_to_local(self.goal_pos)

        # Flatten terrain patch and add additional info
        obs = np.concatenate([
            terrain_patch.flatten(),
            start_local,
            goal_local,
            [float(self.foot_side)]
        ])

        return obs.astype(np.float32)

    def _get_terrain_patch(self) -> np.ndarray:
        """Extract 200x200 terrain patch around current position."""
        x, y = self.position
        half_range = self.obs_range // 2

        # Calculate patch boundaries
        x_start = int(x - half_range)
        y_start = int(y - half_range)

        # Extract patch with boundary handling
        patch = np.zeros((self.obs_range, self.obs_range))

        for i in range(self.obs_range):
            for j in range(self.obs_range):
                global_x = x_start + i
                global_y = y_start + j

                # Check boundaries
                if (0 <= global_x < self.terrain_height and
                    0 <= global_y < self.terrain_width):
                    patch[i, j] = self.terrain.get_height(global_x, global_y)
                else:
                    patch[i, j] = -np.inf  # Minus infinitive height for out-of-bounds

        return patch

    def _generate_terrain_image(self):
        """Generate and save terrain height map as background image."""
        if os.path.exists(self.terrain_image_path):
            return

        grid_size = 0.5
        x_samples = np.arange(0, self.terrain_height, grid_size)
        y_samples = np.arange(0, self.terrain_width, grid_size)

        height_array = np.zeros((len(y_samples), len(x_samples)))
        for i, y in enumerate(y_samples):
            for j, x in enumerate(x_samples):
                height_array[i, j] = self.terrain.get_height(x, y)

        min_height = height_array.min()
        max_height = height_array.max()
        if max_height > min_height:
            normalized = ((height_array - min_height) / (max_height - min_height) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(height_array, dtype=np.uint8)

        # Create RGB image
        rgb_array = np.zeros((len(y_samples), len(x_samples), 3), dtype=np.uint8)

        for i in range(len(y_samples)):
            for j in range(len(x_samples)):
                height = normalized[i, j] / 255.0  # Normalize to 0-1

                if height < 0.33:
                    t = height / 0.33
                    rgb_array[i, j] = [int(60 + 40 * t), int(60 + 40 * t), int(70 + 50 * t)]
                elif height < 0.66:
                    t = (height - 0.33) / 0.33
                    rgb_array[i, j] = [int(100 + 30 * t), int(100 + 20 * t), int(120 - 40 * t)]
                else:
                    t = (height - 0.66) / 0.34
                    rgb_array[i, j] = [int(130 + 50 * t), int(120 + 50 * t), int(80 + 40 * t)]

        # Convert to pygame surface and save
        surface = pygame.surfarray.make_surface(rgb_array.swapaxes(0, 1))
        pygame.image.save(surface, self.terrain_image_path)
        print(f"Generated terrain background image: {self.terrain_image_path}")

    def _global_to_local(self, global_pos: np.ndarray) -> np.ndarray:
        """Transform global coordinates to local frame relative to current position."""
        diff = global_pos - self.position

        # Rotate to local frame
        cos_theta = math.cos(-self.orientation)
        sin_theta = math.sin(-self.orientation)

        local_x = diff[0] * cos_theta - diff[1] * sin_theta
        local_y = diff[0] * sin_theta + diff[1] * cos_theta

        return np.array([local_x, local_y])

    
    def render(self):
        """Simple pygame visualization."""
        if self.render_mode != "human":
            return

        # Initialize pygame once
        if not hasattr(self, '_pygame_initialized'):
            pygame.init()
            self.screen_size = (800, 600)
            self.screen = pygame.display.set_mode(self.screen_size)
            pygame.display.set_caption("Bipedal Robot")
            self.clock = pygame.time.Clock()
            self._pygame_initialized = True

        # Cache terrain surface once
        if not hasattr(self, '_terrain_surface'):
            self._create_terrain_surface()

        # Draw everything
        self.screen.blit(self._terrain_surface, (0, 0))
        self._draw_elements()

        # Handle events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                self._pygame_initialized = False
                return

        pygame.display.flip()
        self.clock.tick(30)  # 30 FPS

    def _create_terrain_surface(self):
        """Create terrain background surface from saved image."""
        # Load terrain image
        if os.path.exists(self.terrain_image_path):
            # Load the terrain image
            terrain_image = pygame.image.load(self.terrain_image_path)

            # Scale the image to fit the coordinate system
            image_width = self.screen_size[0]
            image_height = int(self.terrain_width * self.scale_factor)

            # Scale the terrain image
            scaled_terrain = pygame.transform.scale(terrain_image, (image_width, image_height))

            # Create surface and position terrain image
            self._terrain_surface = pygame.Surface(self.screen_size)
            self._terrain_surface.fill((255, 255, 255))

            # Position: y=500 at middle, so offset by half image height
            terrain_y = (self.screen_size[1] - image_height) // 2

            self._terrain_surface.blit(scaled_terrain, (0, terrain_y))
        else:
            # Fallback to white background
            self._terrain_surface = pygame.Surface(self.screen_size)
            self._terrain_surface.fill((255, 255, 255))

        
    def _draw_elements(self):
        """Draw footsteps, start, goal using proper coordinate transformation."""
        try:
            import pygame
        except ImportError:
            return

        def world_to_screen(x, y):
            """Transform world coordinates to screen coordinates."""
            # x=0 at left edge, x=3000 at right edge, y=500 at middle
            screen_x = (x / 3000.0) * self.screen_size[0]
            screen_y = self.screen_size[1] / 2 - (y - 500) * self.scale_factor
            return (int(screen_x), int(screen_y))

        # Draw footsteps like reference implementation
        for i, pos in enumerate(self.trajectory):
            x, y = world_to_screen(pos[0], pos[1])
            color = (221, 103, 75) if i % 2 == 0 else (75, 164, 221)  # Reference colors
            pygame.draw.circle(self.screen, color, (x, y), 3)  # Draw as circles like reference

        # Draw start position (green circle)
        start = world_to_screen(self.start_pos[0], self.start_pos[1])
        pygame.draw.circle(self.screen, (0, 255, 0), start, 5)

        # Draw goal position (red circle) - only if visible
        goal = world_to_screen(self.goal_pos[0], self.goal_pos[1])
        if 0 <= goal[0] <= self.screen_size[0] and 0 <= goal[1] <= self.screen_size[1]:
            pygame.draw.circle(self.screen, (255, 0, 0), goal, 5)


if __name__ == "__main__":
    # Test environment
    env = BipedEnvironment(render_mode="human")

    print("Testing BipedEnvironment")
    print(f"Action space: {env.action_space}")
    print(f"Observation space shape: {env.observation_space.shape}")

    # Reset environment
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    # Test a few steps
    running = True
    step = 0
    while running and step < 100:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        print(f"Step {step}: reward={reward:.2f}, terminated={terminated}")
        step += 1

        if terminated:
            print("Goal reached!")
            break

        # Small delay to make visualization visible
        pygame.time.wait(100)

    # Keep window open for a moment to see the final state
    if hasattr(env, '_pygame_initialized') and env._pygame_initialized:
        pygame.time.wait(2000)
        pygame.quit()