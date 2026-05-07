import pygame
import random
import time
import sys

class EEGConeGame:
    def __init__(self, time_minutes=1.0, speed=10):
        pygame.init()
        display_info = pygame.display.Info()
        self.width, self.height = display_info.current_w, display_info.current_h
        self.screen = pygame.display.set_mode((self.width, self.height), pygame.FULLSCREEN)
        pygame.display.set_caption("EEG BCI Cone Avoidance")
        self.clock = pygame.time.Clock()
        
        # Game Settings
        self.fps = 24
        self.time_limit_sec = time_minutes * 60
        self.speed = speed
        self.lane_width = self.width / 3
        self.player_width = int(self.lane_width * 0.7)
        self.player_height = int(self.height * 0.12)
        self.obstacle_width = self.player_width
        self.obstacle_height = self.player_height
        self.horizontal_margin = int((self.lane_width - self.player_width) / 2)
        self.vertical_margin = int(self.height * 0.02)
        
        # Colors
        self.BG_COLOR = (40, 44, 52)
        self.PLAYER_COLOR = (97, 175, 239) # Blue
        self.CONE_COLOR = (224, 108, 117)  # Red
        self.LANE_COLOR = (171, 178, 191)  # Grey

        self.reset()

    def reset(self):
        self.start_time = time.time()
        self.player_lane = 1 # Start in middle lane (0: Left, 1: Middle, 2: Right)
        self.obstacles = []  
        self.score = 0
        self.frames_since_spawn = 0
        return self._get_state()

    def _get_state(self):
        return {
            'player_lane': self.player_lane,
            'obstacles': self.obstacles,
            'time_left': max(0, self.time_limit_sec - (time.time() - self.start_time))
        }

    def step(self, action):
        """
        action: 0 (Move Left), 1 (Stay), 2 (Move Right)
        Returns: (state, reward, done)
        """
        reward = 0
        done = False
        
        # 1. Process Input (Relative Movement)
        if action == 0:   # Move Left
            self.player_lane = max(0, self.player_lane - 1)
        elif action == 2: # Move Right
            self.player_lane = min(2, self.player_lane + 1)
        # If action == 1 (Stay), player_lane remains unchanged

        # 2. Spawn Obstacles
        self.frames_since_spawn += 1
        if self.frames_since_spawn > 60: 
            self.frames_since_spawn = 0
            # Ensure at least 1 lane is free
            lanes = [0, 1, 2]
            random.shuffle(lanes)
            num_cones = random.choice([1, 2]) 
            
            for i in range(num_cones):
                self.obstacles.append({'lane': lanes[i], 'y': -self.obstacle_height - self.vertical_margin, 'passed': False})

        # 3. Update Obstacles & Check Collisions
        player_y = self.height - self.player_height - self.vertical_margin
        player_rect = pygame.Rect(
            int(self.player_lane * self.lane_width + self.horizontal_margin),
            player_y,
            self.player_width,
            self.player_height,
        )

        for obs in self.obstacles[:]:
            obs['y'] += self.speed
            obs_rect = pygame.Rect(
                int(obs['lane'] * self.lane_width + self.horizontal_margin),
                int(obs['y']),
                self.obstacle_width,
                self.obstacle_height,
            )

            # Collision Detection
            if player_rect.colliderect(obs_rect):
                reward -= 1
                self.obstacles.remove(obs)
            # Passed Detection
            elif obs['y'] > player_y + 100 and not obs['passed']:
                obs['passed'] = True
            # Clean up off-screen obstacles
            elif obs['y'] > self.height:
                self.obstacles.remove(obs)

        # 4. Check Timer
        if (time.time() - self.start_time) >= self.time_limit_sec:
            done = True

        return self._get_state(), reward, done

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.screen.fill(self.BG_COLOR)

        # Draw Lanes
        for i in range(1, 3):
            pygame.draw.line(self.screen, self.LANE_COLOR, (i * self.lane_width, 0), (i * self.lane_width, self.height), 2)

        # Draw Player
        player_x = int(self.player_lane * self.lane_width + self.horizontal_margin)
        player_y = self.height - self.player_height - self.vertical_margin
        pygame.draw.rect(
            self.screen,
            self.PLAYER_COLOR,
            (player_x, player_y, self.player_width, self.player_height),
        )

        # Draw Obstacles
        for obs in self.obstacles:
            obs_x = int(obs['lane'] * self.lane_width + self.horizontal_margin)
            pygame.draw.polygon(
                self.screen,
                self.CONE_COLOR,
                [
                    (obs_x + self.obstacle_width // 2, obs['y']),
                    (obs_x, obs['y'] + self.obstacle_height),
                    (obs_x + self.obstacle_width, obs['y'] + self.obstacle_height),
                ],
            )

        # Draw Score and Timer
        font = pygame.font.SysFont(None, 36)
        time_left = int(max(0, self.time_limit_sec - (time.time() - self.start_time)))
        score_text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        time_text = font.render(f"Time: {time_left}s", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))
        self.screen.blit(time_text, (self.width - 120, 10))

        pygame.display.flip()
        self.clock.tick(self.fps)



