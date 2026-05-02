import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
import torch.nn
import torch
import torch.optim as optim

P_FREE = 0.06
MIN_SERVERS = 6
TOTAL_SERVERS = 10
QUEUE = [1, 2, 4, 8]
rng = np.random.default_rng(0)
alpha = 0.01
beta = 0.01
epsilon = 0.1

class AccessControlQueue(gym.Env):
    def __init__(self, num_servers, queue):
        self.num_servers = num_servers
        self.queue = queue
        self._free_servers = 0
        self._queue_head = 0
        self.action_space = Discrete(2, seed=42)

    def _get_obs(self):
        return {"Available Servers": self.free_servers, "Queue Head": self.queue_head}

    def _get_info(self):
        return {}

    def reset(self):
        self.queue_head = rng.choice(self.queue, size=1)[0]
        self.free_servers = rng.integers(MIN_SERVERS, self.num_servers)
        observation = self._get_obs()
        return observation

    def step(self, action):
        reward = 0
        if action == 1 and self.free_servers > 0:
            self.free_servers -= 1
            reward = self.queue_head
        elif action == 1 and self.free_servers == 0:
            reward = 0
        elif action == 0:
            reward = 0

        if self.free_servers < self.num_servers:
            busy_servers = self.num_servers - self.free_servers
            self.free_servers += rng.binomial(busy_servers, P_FREE)

        self.queue_head = rng.choice(self.queue, size=1)[0]

        observation = self._get_obs()
        truncated = False
        terminated = False
        info = self._get_info()
        return observation, reward, terminated, truncated, info


class QValue(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(3, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, 1)

    def forward(self, x):
        x =torch.relu(self.fc1(x))
        x =torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x



def approximateQ(model, obs, action):
    model_in = torch.tensor([obs["Available Servers"]/TOTAL_SERVERS, obs["Queue Head"]/8, action], dtype=torch.float32)
    model_out = model(model_in)
    return model_out


def epsilon_greedy(Q, obs, epsilon):
    if rng.random() < epsilon:
        return env.action_space.sample()
    else:
        q0 = approximateQ(model, obs, 0).detach()
        q1 = approximateQ(model, obs, 1).detach()
        q_vals = torch.stack([q0, q1])
        return int(torch.argmax(q_vals).item())



env = AccessControlQueue(TOTAL_SERVERS, queue=QUEUE)
R_hat = 0
model = QValue()
optimizer = optim.Adam(model.parameters(), lr=0.001)
steps = 200000
obs = env.reset()
action = env.action_space.sample()
reward_history = []
rhat_history = []

for i in range (1,steps):
    print("Step: ", i)
    obs_next, reward, _, _, _ = env.step(action)
    action_next = epsilon_greedy(model, obs_next, epsilon)
    q_sa = approximateQ(model, obs, action)
    q_snext_anext = approximateQ(model, obs_next, action_next).detach()
    delta = reward - R_hat + q_snext_anext - q_sa
    R_hat = R_hat + beta * delta.item()
    loss = -delta.detach() * q_sa
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    obs = obs_next
    action = action_next
    reward_history.append(reward)
    rhat_history.append(R_hat)


def nn_greedy(state):
    fs, p = state
    obs = {"Available Servers": fs, "Queue Head": p}
    with torch.no_grad():
        q = [approximateQ(model, obs, a).item() for a in (0, 1)]
    return int(np.argmax(q))

def print_policy_table(policy_fn, label=""):
    print(f"\n[1=ACCEPT, 0=REJECT]")
    header = "  free_servers ->" + "".join(f" {fs:2d}" for fs in range(TOTAL_SERVERS + 1))
    print(header)
    for p in QUEUE:
        row = f"  priority {p:<2}     "
        for fs in range(TOTAL_SERVERS + 1):
            row += f"  {policy_fn((fs, p))}"
        print(row)

print_policy_table(nn_greedy, label="NN model")
