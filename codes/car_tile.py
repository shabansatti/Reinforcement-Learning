import numpy as np
import gymnasium as gym

from utils.tiles import tiles, IHT

rng = np.random.default_rng(0)
alpha = 0.01
beta = 0.01
epsilon = 0.1
# --------------------------------------------------------------------------
# Mountain Car constants (same scaling as book)
# --------------------------------------------------------------------------
POS_MIN, POS_MAX = -1.2, 0.5
VEL_MIN, VEL_MAX = -0.07, 0.07

ACTIONS = (0, 1, 2)   # Gymnasium: 0=left, 1=neutral, 2=right

# --------------------------------------------------------------------------
# Tile coding setup
# --------------------------------------------------------------------------
NUM_TILINGS = 8
IHT_SIZE    = 4096
POS_SCALE   = NUM_TILINGS / (POS_MAX - POS_MIN)
VEL_SCALE   = NUM_TILINGS / (VEL_MAX - VEL_MIN)

def active_tiles(iht, pos, vel, action):
    return tiles(iht,
                 NUM_TILINGS,
                 [POS_SCALE * pos, VEL_SCALE * vel],
                 [action])

def q_hat(w, tile_ids):
    return w[tile_ids].sum()

def greedy_action(w, iht, pos, vel):
    values = [q_hat(w, active_tiles(iht, pos, vel, a)) for a in ACTIONS]
    best   = np.flatnonzero(values == np.max(values))
    return ACTIONS[np.random.choice(best)]




env = gym.make("MountainCar-v0")
alpha = 0.5 / NUM_TILINGS
episodes = 2000
reward_history = []
n = 4
gamma = 1
iht = IHT(IHT_SIZE)
w = np.zeros(IHT_SIZE)
for episode in range (1,episodes):
    t = 0
    Tau = 0
    T = np.inf
    R = []
    S = []
    A = []
    ep_return = 0.0
    obs, _ = env.reset()
    action = env.action_space.sample()
    S.append(obs)
    A.append(action)
    R.append(0)
    epsilon = max(0.05, 1.0 - 0.95 * episode / 500)

    while True:
        if t < T:
            obs_next, reward, terminated, truncated, _ = env.step(action)
            pos_next, vel_next = obs_next
            ep_return += reward
            R.append(reward)
            S.append(obs_next)
            if terminated or truncated:
                T = t + 1
            else:
                action_next = greedy_action(w, iht, pos_next, vel_next)
                action = action_next
                A.append(action_next)
        Tau = t -  n + 1
        if Tau >= 0:
            G = 0.0
            upper = min(Tau + n, T)

            for i in range(Tau + 1, upper + 1):
                G += (gamma ** (i - Tau - 1)) * R[i]

            if Tau + n < T:
                tiles_nn = active_tiles(iht,
                                        S[Tau + n][0],
                                        S[Tau + n][1],
                                        A[Tau + n])
                G += (gamma ** n) * q_hat(w, tiles_nn)

            tiles_tau = active_tiles(iht,
                                     S[Tau][0],
                                     S[Tau][1],
                                     A[Tau])

            w[tiles_tau] += alpha * (G - q_hat(w, tiles_tau))

        if (Tau == T-1):
            break
        t += 1
    reward_history.append(ep_return)
    if episode % 10 == 0:
        avg_return = np.mean(reward_history[-10:])
        print(f"episode {episode:5d} | avg return (last 10): {avg_return:7.1f}")

eval_env = gym.make("MountainCar-v0", render_mode="human")
for ep in range(5):
    obs, _ = eval_env.reset()
    total_reward = 0.0
    steps = 0
    done = False
    while not done:
        action = greedy_action(w, iht, obs[0], obs[1])
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated
    print(f"Eval {ep+1}: steps={steps}, return={total_reward:.0f}, "
          f"reached_flag={terminated}")
eval_env.close()