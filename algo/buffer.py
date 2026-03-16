import random
import numpy as np
import os
import pickle
from pathlib import Path

class ReplayMemory:
    def __init__(self, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, safety, action, reward, next_state, next_safety, mask):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, safety, action, reward, next_state, next_safety, mask)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, safety, action, reward, next_state, next_safety, mask = map(lambda x: np.stack(x).astype(np.float32), zip(*batch))
        return state, safety, action, reward, next_state, next_safety, mask

    def __len__(self):
        return len(self.buffer)

    def save_buffer(self, data_name, save_path=None):
        dir_path = f"data/{data_name}_replay_buffer/"
        os.makedirs(dir_path, exist_ok=True)

        if save_path is None:
            file_name = f"{data_name}_buffer.pkl"
            save_path = os.path.join(dir_path, file_name)

        print(f"[✓] Saving buffer to {save_path}")
        with open(save_path, 'wb') as f:
            pickle.dump(self.buffer, f)

    def load_buffer(self, data_name_or_path):
        if os.path.isdir(data_name_or_path):
            file_path = os.path.join(data_name_or_path, f"{Path(data_name_or_path).name.replace('_replay_buffer', '')}_buffer.pkl")
        else:
            file_path = data_name_or_path

        print(f"[✓] Loading buffer from {file_path}")
        with open(file_path, "rb") as f:
            self.buffer = pickle.load(f)
            self.position = len(self.buffer) % self.capacity