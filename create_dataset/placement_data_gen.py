import numpy as np
import random
import pickle
import os
from collections import defaultdict

# Define chip dimensions
GRID_H        = 100 #height
GRID_W        = 100 #width
NUM_MACROS    = 10  #number of macros / episode
NUM_EPISODES  = 1000 # number of random trajectories
ALPHA         = 1.0  # HPWL weight
BETA          = 0.5  # congestion (RUDY) weight
SAVE_PATH     = "trajectories.pkl"

random.seed(42)
np.random.seed(42)


# Generate randomly sized macros
# structure:{id, w, h}
def generate_macros(num_macros, grid_h, grid_w):
    macros = []
    for i in range(num_macros):
        w = random.randint(5, 15)
        h = random.randint(5, 15)
        macros.append({"id": i, "w": w, "h": h})
    return macros

# Generate random netlist
# CHANGE TO IBM DATA LATER
# each macro connects 2-4 others
def generate_netlist(num_macros, num_nets=15):
    netlist = []
    for _ in range(num_nets):
        size = random.randint(2, min(4, num_macros))
        net  = random.sample(range(num_macros), size)
        netlist.append(net)
    return netlist


# Placements: Randomly sample positions until legal one is found. Otherwise return (x,y)
def get_legal_position(grid, macro_w, macro_h, max_attempts=1000):
    grid_h, grid_w = grid.shape
    # if macro bigger than grid then return None
    if macro_w > grid_w or macro_h > grid_h:
        return None
    
    for _ in range(max_attempts):
        x = random.randint(0, grid_w - macro_w)
        y = random.randint(0, grid_h - macro_h)
        region = grid[y:y + macro_h, x:x + macro_w]
        if np.all(region == 0):
            return x, y
    # no legal placement found
    return None  

# add macro to grid
def place_macro(grid, macro_w, macro_h, x, y):
    g = grid.copy()
    g[y:y + macro_h, x:x + macro_w] = 1
    return g


#Reward functions 
#HPWL
def compute_hpwl(placed_positions, macros, netlist):
    #build center positions for each placed macro
    centers = {}
    for macro_id, x, y in placed_positions:
        mw = macros[macro_id]["w"]
        mh = macros[macro_id]["h"]
        cx = x + mw / 2.0
        cy = y + mh / 2.0
        centers[macro_id] = (cx, cy)

    hpwl = 0.0
    for net in netlist:
        # only consider macros that are actually placed
        net_placed = [m for m in net if m in centers]
        if len(net_placed) < 2:
            continue
        xs = [centers[m][0] for m in net_placed]
        ys = [centers[m][1] for m in net_placed]
        hpwl += (max(xs) - min(xs)) + (max(ys) - min(ys))
    return hpwl

#RUDY function
def compute_rudy(placed_positions, macros, netlist, grid_h, grid_w, bin_size=10):
    bins_h = grid_h // bin_size
    bins_w = grid_w // bin_size
    density = np.zeros((bins_h, bins_w))

    centers = {}
    for macro_id, x, y in placed_positions:
        mw = macros[macro_id]["w"]
        mh = macros[macro_id]["h"]
        centers[macro_id] = (x + mw / 2.0, y + mh / 2.0)

    for net in netlist:
        net_placed = [m for m in net if m in centers]
        if len(net_placed) < 2:
            continue
        xs = [centers[m][0] for m in net_placed]
        ys = [centers[m][1] for m in net_placed]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bbox_area = max((x_max - x_min) * (y_max - y_min), 1.0)
        wire_len  = (x_max - x_min) + (y_max - y_min)
        rudy_val  = wire_len / bbox_area

        # find which bins this net's bbox overlaps
        bx_min = int(x_min // bin_size)
        bx_max = int(x_max // bin_size)
        by_min = int(y_min // bin_size)
        by_max = int(y_max // bin_size)
        bx_min = max(0, min(bx_min, bins_w - 1))
        bx_max = max(0, min(bx_max, bins_w - 1))
        by_min = max(0, min(by_min, bins_h - 1))
        by_max = max(0, min(by_max, bins_h - 1))

        density[by_min:by_max+1, bx_min:bx_max+1] += rudy_val

    return float(np.mean(density))

#Reward function 
def compute_reward(placed_positions, macros, netlist, grid_h, grid_w, alpha=ALPHA, beta=BETA):

    hpwl = compute_hpwl(placed_positions, macros, netlist)
    rudy = compute_rudy(placed_positions, macros, netlist, grid_h, grid_w)
    reward = -(alpha * hpwl + beta * rudy)
    return reward, hpwl, rudy


#Episodes
#Run one episode, return trajectory of snapshot and final reward
#if no placemennt found, return none 
def run_episode(macros, netlist, grid_h, grid_w):
    grid      = np.zeros((grid_h, grid_w), dtype=np.float32)
    remaining = list(range(len(macros)))
    random.shuffle(remaining) 
    placed    = []             
    snapshots = []

    for macro_id in remaining:
        mw = macros[macro_id]["w"]
        mh = macros[macro_id]["h"]

        pos = get_legal_position(grid, mw, mh)
        if pos is None:
            return None  #skip episode if grid is full

        x, y = pos

        # save snapshot before placing
        snapshots.append({
            "step":      len(placed),
            "grid":      grid.copy(),          # (H, W) occupancy grid
            "placed":    placed.copy(),         # [(macro_id, x, y), ...]
            "action":    (macro_id, x, y),      # action taken
            "remaining": [m for m in remaining if m not in [p[0] for p in placed] and m != macro_id],
        })

        grid   = place_macro(grid, mw, mh, x, y)
        placed.append((macro_id, x, y))

    reward, hpwl, rudy = compute_reward(placed, macros, netlist, grid_h, grid_w)

    return {
        "snapshots":      snapshots,
        "final_grid":     grid,
        "placed":         placed,
        "reward":         reward,
        "hpwl":           hpwl,
        "rudy":           rudy,
    }


#Generate dataset
def generate_dataset(num_episodes, macros, netlist, grid_h, grid_w):
    trajectories = []
    failed        = 0

    for ep in range(num_episodes):
        traj = run_episode(macros, netlist, grid_h, grid_w)
        if traj is None:
            failed += 1
            continue
        trajectories.append(traj)

        if (ep + 1) % 100 == 0:
            print(f"  Episode {ep+1}/{num_episodes} | "
                  f"collected: {len(trajectories)} | failed: {failed}")

    print(f"\nDone. {len(trajectories)} trajectories collected, {failed} failed.")
    return trajectories


#Label trajectories
def label_trajectories(trajectories):
    rewards = np.array([t["reward"] for t in trajectories])
    r_min, r_max = rewards.min(), rewards.max()

    # normalize to [0, 1]
    norm = (rewards - r_min) / (r_max - r_min + 1e-8)

    q75 = np.percentile(norm, 75)
    q25 = np.percentile(norm, 25)

    for i, traj in enumerate(trajectories):
        traj["score"] = float(norm[i])
        if norm[i] >= q75:
            traj["label"] = "good"
        elif norm[i] <= q25:
            traj["label"] = "bad"
        else:
            traj["label"] = "medium"

    return trajectories


#save dataset
def save_dataset(trajectories, macros, netlist, path=SAVE_PATH):
    dataset = {
        "macros":       macros,
        "netlist":      netlist,
        "grid_h":       GRID_H,
        "grid_w":       GRID_W,
        "trajectories": trajectories,
    }
    with open(path, "wb") as f:
        pickle.dump(dataset, f)
    size_mb = os.path.getsize(path) / 1e6
    print(f"Dataset saved to '{path}' ({size_mb:.1f} MB)")


def load_dataset(path=SAVE_PATH):
    with open(path, "rb") as f:
        return pickle.load(f)


#quick stats
def print_stats(trajectories):
    rewards = [t["reward"] for t in trajectories]
    labels  = defaultdict(int)
    for t in trajectories:
        labels[t["label"]] += 1

    print("\n─── Dataset Stats ───────────────────────")
    print(f"  Total trajectories : {len(trajectories)}")
    print(f"  Reward  min        : {min(rewards):.2f}")
    print(f"  Reward  max        : {max(rewards):.2f}")
    print(f"  Reward  mean       : {np.mean(rewards):.2f}")
    print(f"  Labels  good       : {labels['good']}")
    print(f"  Labels  medium     : {labels['medium']}")
    print(f"  Labels  bad        : {labels['bad']}")
    print("─────────────────────────────────────────\n")


#main running program
if __name__ == "__main__":
    print("Generating macros and netlist...")
    macros  = generate_macros(NUM_MACROS, GRID_H, GRID_W)
    netlist = generate_netlist(NUM_MACROS)

    print(f"Macros  : {macros}")
    print(f"Netlist : {netlist}\n")

    print(f"Running {NUM_EPISODES} episodes...")
    trajectories = generate_dataset(NUM_EPISODES, macros, netlist, GRID_H, GRID_W)

    print("Labeling trajectories...")
    trajectories = label_trajectories(trajectories)

    print_stats(trajectories)

    save_dataset(trajectories, macros, netlist)

    # ── sanity check: peek at one trajectory ──
    t0 = trajectories[0]
    print(f"Sample trajectory:")
    print(f"  Steps      : {len(t0['snapshots'])}")
    print(f"  Reward     : {t0['reward']:.4f}")
    print(f"  HPWL       : {t0['hpwl']:.4f}")
    print(f"  RUDY       : {t0['rudy']:.4f}")
    print(f"  Label      : {t0['label']}")
    print(f"  Grid shape : {t0['final_grid'].shape}")
    print(f"  Actions    : {t0['placed']}")