import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2

N_TYPES = 11
GRAY_STEP = 255 // (N_TYPES + 1)
BG_GRAY = 0
GRAYS = [n * GRAY_STEP for n in range(1, N_TYPES + 1)]
assert len(GRAYS) == N_TYPES

SRC_BOARD_OFFSET = (30, 31)
SRC_WALL_HEIGHT_OFFSET = 130
SRC_BOARD_CROPPED_SIZE = (709, 508)
WALL_THICKNESS = 4


class CoordSizeToImage(gym.Wrapper):
    def __init__(self, env, image_size=(96, 96)):
        gym.ObservationWrapper.__init__(self, env)
        self.n_frames = env.unwrapped.n_frames
        self.image_size = image_size
        self.resize_ratio = (96 - WALL_THICKNESS) / SRC_BOARD_CROPPED_SIZE[0]
        board_width = (
            round(SRC_BOARD_CROPPED_SIZE[1] * self.resize_ratio) + WALL_THICKNESS * 2
        )  # 74
        horizontal_offset = (image_size[1] - board_width) // 2
        self.horizontal_wall_thickness = horizontal_offset + WALL_THICKNESS
        # [0, (horizontal_wall_thickness - 1)] for wall
        self.fruit_horizontal_offset = self.horizontal_wall_thickness
        self.wall_height_offset = round(SRC_WALL_HEIGHT_OFFSET * self.resize_ratio)
        self.observation_space = spaces.Dict(
            {
                "boards": spaces.Box(
                    low=0,
                    high=255,
                    shape=(
                        self.n_frames,  # [C, H, W]
                        *image_size,
                    ),
                    dtype=np.uint8,
                ),
                "cur_fruit": spaces.Discrete(5),
                "next_fruit": spaces.Discrete(5),
            }
        )

        self.max_depth = self.image_size[0] - self.wall_height_offset - WALL_THICKNESS
        self.prev_min_depth = self.max_depth
        self.prev_mean_depth = self.max_depth

    def _transform(self, board):
        return [
            (
                (
                    round((pos[0] - SRC_BOARD_OFFSET[0]) * self.resize_ratio)
                    + self.fruit_horizontal_offset,
                    round((pos[1] - SRC_BOARD_OFFSET[1]) * self.resize_ratio),
                ),
                r * self.resize_ratio,
                t,
            )
            for pos, r, t in board
        ]

    def observation(self, observation):
        images = []
        for board in observation["boards"]:
            image = np.zeros(self.image_size, dtype=np.uint8)
            # left wall
            cv2.rectangle(
                image,
                (0, self.wall_height_offset),
                (self.horizontal_wall_thickness - 1, self.image_size[0] - 1),
                255,
                -1,
            )
            # right wall
            cv2.rectangle(
                image,
                (
                    self.image_size[1] - self.horizontal_wall_thickness,
                    self.wall_height_offset,
                ),
                (self.image_size[1] - 1, self.image_size[0] - 1),
                255,
                -1,
            )
            # bottom wall
            cv2.rectangle(
                image,
                (0, self.image_size[0] - WALL_THICKNESS),
                (self.image_size[1], self.image_size[0] - 1),
                255,
                -1,
            )
            for pos, r, t in self._transform(board):
                cv2.circle(
                    image, center=pos, radius=int(r), color=GRAYS[t], thickness=-1
                )
            images.append(image)

        observation["boards"] = (
            np.array(images)
            .reshape(self.n_frames, *self.image_size)  # [C,H,W]
            .astype(np.uint8)
        )

        # SB3 will handle conversion to tensor

        return observation

    def shape_reward(self, obs, reward, info):
        board = obs["boards"][-1]
        first_nonzeros = np.argmax(board.T != 0, axis=-1)
        depths = first_nonzeros - self.wall_height_offset
        depths = depths[
            self.fruit_horizontal_offset : (
                self.image_size[1] - self.horizontal_wall_thickness - 1
            )
        ]
        min_depth = np.min(depths)
        mean_depth = np.mean(depths)

        min_depth_delta = min_depth - self.prev_min_depth
        mean_depth_delta = mean_depth - self.prev_mean_depth

        self.prev_min_depth = min_depth
        self.prev_mean_depth = mean_depth

        # print(
        #    f"Depths: {depths}, Min: {min_depth}, Mean: {mean_depth}, "
        #    f"Delta: {min_depth_delta:.2f}, {mean_depth_delta:.2f}" # )
        merge_count = info["merge_count"]
        shaped_reward = (
            merge_count / 4  # hyperparameter, can be tuned
            + min_depth_delta / self.max_depth * 2
            + mean_depth_delta / self.max_depth * 4
        )
        # print(f"Shaped reward: {shaped_reward:.2f} (merge_count: {merge_count})")
        return shaped_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        obs = self.observation(obs)
        shaped_reward = self.shape_reward(obs, reward, info)

        return obs, shaped_reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.observation(obs)
        self.prev_min_depth = self.max_depth
        self.prev_mean_depth = self.max_depth
        return obs, info
