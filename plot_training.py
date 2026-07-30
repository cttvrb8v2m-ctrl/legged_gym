import argparse
import os
import glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['lines.linewidth'] = 1.5

COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
    '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
    '#bcbd22', '#17becf'
]


def load_tensorboard_data(log_dir):
    event_files = glob.glob(os.path.join(log_dir, 'events.out.tfevents.*'))
    if not event_files:
        raise ValueError(f"No TensorBoard event files found in {log_dir}")
    
    event_file = event_files[0]
    accumulator = EventAccumulator(event_file, size_guidance={
        'scalars': 0,
        'histograms': 0,
        'images': 0,
        'audio': 0,
        'tensors': 0
    })
    accumulator.Reload()
    
    tags = accumulator.Tags()['scalars']
    data = {}
    for tag in tags:
        events = accumulator.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        data[tag] = {'steps': steps, 'values': values}
    
    return data


def smooth_curve(values, window_size=10):
    if len(values) < window_size:
        return values
    kernel = np.ones(window_size) / window_size
    return np.convolve(values, kernel, mode='same')


def plot_reward(data, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'Train/mean_reward' in data:
        steps = data['Train/mean_reward']['steps']
        values = smooth_curve(data['Train/mean_reward']['values'])
        ax.plot(steps, values, color=COLORS[0], label='Mean Reward')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Training Reward Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'reward.png'), bbox_inches='tight')
    plt.close()


def plot_episode_length(data, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'Train/mean_episode_length' in data:
        steps = data['Train/mean_episode_length']['steps']
        values = smooth_curve(data['Train/mean_episode_length']['values'])
        ax.plot(steps, values, color=COLORS[1], label='Mean Episode Length')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Episode Length')
    ax.set_title('Episode Length Curve')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'episode_length.png'), bbox_inches='tight')
    plt.close()


def plot_velocity_rewards(data, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    vel_tags = [
        ('Episode/rew_tracking_lin_vel', 'Linear Velocity'),
        ('Episode/rew_tracking_ang_vel', 'Angular Velocity')
    ]
    for i, (tag, label) in enumerate(vel_tags):
        if tag in data:
            steps = data[tag]['steps']
            values = smooth_curve(data[tag]['values'])
            ax.plot(steps, values, color=COLORS[i], label=label)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward Value')
    ax.set_title('Velocity Tracking Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'velocity.png'), bbox_inches='tight')
    plt.close()


def plot_gait_rewards(data, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    gait_tags = [
        ('Episode/rew_feet_width', 'Feet Width'),
        ('Episode/rew_feet_distance', 'Feet Distance'),
        ('Episode/rew_feet_air_time', 'Feet Air Time')
    ]
    for i, (tag, label) in enumerate(gait_tags):
        if tag in data:
            steps = data[tag]['steps']
            values = smooth_curve(data[tag]['values'])
            ax.plot(steps, values, color=COLORS[i], label=label)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward Value')
    ax.set_title('Gait-related Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'gait_reward.png'), bbox_inches='tight')
    plt.close()


def plot_stability_rewards(data, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    stability_tags = [
        ('Episode/rew_orientation', 'Orientation'),
        ('Episode/rew_collision', 'Collision'),
        ('Episode/rew_lin_vel_z', 'Vertical Velocity')
    ]
    for i, (tag, label) in enumerate(stability_tags):
        if tag in data:
            steps = data[tag]['steps']
            values = smooth_curve(data[tag]['values'])
            ax.plot(steps, values, color=COLORS[i], label=label)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward Value')
    ax.set_title('Stability Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'stability.png'), bbox_inches='tight')
    plt.close()


def plot_loss(data, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    loss_tags = [
        ('Loss/value_function', 'Value Function Loss'),
        ('Loss/surrogate', 'Surrogate Loss')
    ]
    for i, (tag, label) in enumerate(loss_tags):
        if tag in data:
            steps = data[tag]['steps']
            values = smooth_curve(data[tag]['values'])
            ax.plot(steps, values, color=COLORS[i], label=label)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'loss.png'), bbox_inches='tight')
    plt.close()


def plot_exploration(data, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'Policy/mean_noise_std' in data:
        steps = data['Policy/mean_noise_std']['steps']
        values = smooth_curve(data['Policy/mean_noise_std']['values'])
        ax.plot(steps, values, color=COLORS[0], label='Mean Action Noise Std')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Noise Std')
    ax.set_title('Exploration Level')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'exploration.png'), bbox_inches='tight')
    plt.close()


def plot_terrain(data, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    if 'Episode/terrain_level' in data:
        steps = data['Episode/terrain_level']['steps']
        values = smooth_curve(data['Episode/terrain_level']['values'])
        ax.plot(steps, values, color=COLORS[0], label='Terrain Level')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Terrain Level')
    ax.set_title('Terrain Training Progress')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'terrain.png'), bbox_inches='tight')
    plt.close()


def plot_all_rewards(data, save_path):
    reward_tags = [tag for tag in data.keys() if tag.startswith('Episode/rew_')]
    if not reward_tags:
        return
    
    fig, ax = plt.subplots(figsize=(12, 8))
    for i, tag in enumerate(sorted(reward_tags)):
        label = tag.replace('Episode/rew_', '')
        steps = data[tag]['steps']
        values = smooth_curve(data[tag]['values'])
        ax.plot(steps, values, color=COLORS[i % len(COLORS)], label=label)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward Value')
    ax.set_title('All Reward Components')
    ax.legend(ncol=3, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'all_rewards.png'), bbox_inches='tight')
    plt.close()


def plot_multiple_runs(log_dirs, labels, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    plots = [
        ('Train/mean_reward', 'Mean Reward', axes[0, 0]),
        ('Train/mean_episode_length', 'Episode Length', axes[0, 1]),
        ('Loss/value_function', 'Value Function Loss', axes[1, 0]),
        ('Policy/mean_noise_std', 'Noise Std', axes[1, 1])
    ]
    
    for tag, title, ax in plots:
        for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
            try:
                data = load_tensorboard_data(log_dir)
                if tag in data:
                    steps = data[tag]['steps']
                    values = smooth_curve(data[tag]['values'])
                    ax.plot(steps, values, color=COLORS[i], label=label)
            except Exception as e:
                print(f"Failed to load {log_dir}: {e}")
        ax.set_xlabel('Training Steps')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'comparison.png'), bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot training results from TensorBoard logs')
    parser.add_argument('--log_dir', type=str, nargs='+', required=True, help='Path(s) to training log directory')
    parser.add_argument('--labels', type=str, nargs='+', default=None, help='Labels for multiple runs')
    parser.add_argument('--save_dir', type=str, default=None, help='Output directory for plots')
    args = parser.parse_args()
    
    log_dirs = args.log_dir
    labels = args.labels if args.labels else [os.path.basename(os.path.normpath(d)) for d in log_dirs]
    
    if len(log_dirs) == 1:
        log_dir = log_dirs[0]
        if args.save_dir:
            save_path = args.save_dir
        else:
            save_path = os.path.join(log_dir, 'plots')
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Loading data from {log_dir}...")
        data = load_tensorboard_data(log_dir)
        print(f"Found {len(data)} tags")
        
        print("Generating plots...")
        plot_reward(data, save_path)
        plot_episode_length(data, save_path)
        plot_velocity_rewards(data, save_path)
        plot_gait_rewards(data, save_path)
        plot_stability_rewards(data, save_path)
        plot_loss(data, save_path)
        plot_exploration(data, save_path)
        plot_terrain(data, save_path)
        plot_all_rewards(data, save_path)
        
        print(f"Plots saved to {save_path}")
    
    else:
        if args.save_dir:
            save_path = args.save_dir
        else:
            save_path = os.path.join(os.path.dirname(log_dirs[0]), 'comparison_plots')
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Comparing {len(log_dirs)} runs...")
        plot_multiple_runs(log_dirs, labels, save_path)
        print(f"Comparison plot saved to {save_path}")


if __name__ == '__main__':
    main()
