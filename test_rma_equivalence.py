"""Test RMA equivalence with original PPO when alpha=0"""
import torch
import os

# Add paths
import sys
sys.path.insert(0, '/home/gjt/legged_gym')
sys.path.insert(0, '/home/gjt/isaacgym/python/examples/rsl_rl')

from rsl_rl.modules import ActorCritic
from legged_gym.algorithms.rma_actor_critic import RMAActorCritic

def test_rma_equivalence():
    print("=" * 80)
    print("📌 RMA Equivalence Test")
    print("=" * 80)
    
    # Configuration
    num_obs = 235
    num_actions = 12
    history_len = 10
    latent_dim = 32
    batch_size = 64
    
    # Load a PPO checkpoint
    checkpoint_path = '/home/gjt/legged_gym/logs/rough_go1/1/model_550.pt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"\n📁 Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model state dict
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
    elif 'actor_critic' in checkpoint:
        model_state_dict = checkpoint['actor_critic']
    else:
        model_state_dict = checkpoint
    
    print(f"📊 Checkpoint parameters: {len(model_state_dict)}")
    
    # Create original PPO ActorCritic
    print("\n🔧 Creating original ActorCritic...")
    ppo_ac = ActorCritic(num_obs, num_obs, num_actions, 
                        actor_hidden_dims=[512, 256, 128],
                        critic_hidden_dims=[512, 256, 128],
                        activation='elu', init_noise_std=1.0)
    
    # Load weights
    ppo_ac.load_state_dict(model_state_dict, strict=True)
    ppo_ac.eval()
    print("✅ Original ActorCritic loaded")
    
    # Create RMA ActorCritic
    print("\n🔧 Creating RMAActorCritic...")
    rma_ac = RMAActorCritic(num_obs, num_obs, num_actions,
                           history_len=history_len,
                           latent_dim=latent_dim,
                           actor_hidden_dims=[512, 256, 128],
                           critic_hidden_dims=[512, 256, 128],
                           activation='elu', init_noise_std=1.0)
    
    # Load only actor/critic/std weights (RMA modules are initialized to zero)
    filtered_state_dict = {}
    for key, val in model_state_dict.items():
        if key.startswith('actor.') or key.startswith('critic.') or key.startswith('std'):
            if key in rma_ac.state_dict() and rma_ac.state_dict()[key].shape == val.shape:
                filtered_state_dict[key] = val
    
    rma_ac.load_state_dict(filtered_state_dict, strict=False)
    rma_ac.eval()
    
    # Set alpha=0 for identity mapping
    rma_ac.set_rma_alpha(0.0)
    print(f"✅ RMAActorCritic loaded with alpha={rma_ac.rma_alpha}")
    print(f"📊 Filtered parameters loaded: {len(filtered_state_dict)}")
    
    # Generate test observations
    print("\n🔍 Generating test observations...")
    torch.manual_seed(42)
    obs = torch.randn(batch_size, num_obs)
    obs_history = torch.randn(batch_size, history_len, num_obs)
    
    # Get action_mean from original PPO
    print("\n🔹 Original PPO forward pass...")
    ppo_ac.act(obs)
    ppo_action_mean = ppo_ac.action_mean.detach().clone()
    ppo_value = ppo_ac.evaluate(obs).detach().clone()
    print(f"  PPO action_mean shape: {ppo_action_mean.shape}")
    print(f"  PPO action_mean[0]: {ppo_action_mean[0][:3].tolist()}")
    
    # Get action_mean from RMA (with alpha=0)
    print("\n🔹 RMA forward pass (alpha=0)...")
    rma_ac.update_distribution(obs, obs_history)
    rma_action_mean = rma_ac.action_mean.detach().clone()
    rma_value = rma_ac.evaluate(obs, obs_history).detach().clone()
    print(f"  RMA action_mean shape: {rma_action_mean.shape}")
    print(f"  RMA action_mean[0]: {rma_action_mean[0][:3].tolist()}")
    
    # Check RMA stats
    rma_stats = rma_ac.get_rma_stats()
    print(f"\n  RMA stats:")
    print(f"    rma_alpha: {rma_stats.get('rma_alpha', 0.0)}")
    print(f"    latent_max_abs: {rma_stats.get('latent_max_abs', 'N/A')}")
    print(f"    gamma_raw_max_abs: {rma_stats.get('gamma_raw_max_abs', 'N/A')}")
    print(f"    beta_raw_max_abs: {rma_stats.get('beta_raw_max_abs', 'N/A')}")
    
    # Compute differences
    print("\n📊 Comparing outputs...")
    action_diff = torch.abs(ppo_action_mean - rma_action_mean)
    value_diff = torch.abs(ppo_value - rma_value)
    
    print(f"\n  Action Mean Difference:")
    print(f"    Max abs diff: {action_diff.max().item():.8f}")
    print(f"    Mean abs diff: {action_diff.mean().item():.8f}")
    print(f"    Std: {action_diff.std().item():.8f}")
    
    print(f"\n  Value Difference:")
    print(f"    Max abs diff: {value_diff.max().item():.8f}")
    print(f"    Mean abs diff: {value_diff.mean().item():.8f}")
    
    # Check equivalence
    max_action_diff = action_diff.max().item()
    max_value_diff = value_diff.max().item()
    threshold = 1e-5
    
    print("\n" + "=" * 80)
    if max_action_diff < threshold and max_value_diff < threshold:
        print(f"✅ PASSED - RMA with alpha=0 is equivalent to original PPO")
        print(f"   Max action diff: {max_action_diff:.8f} < {threshold}")
        print(f"   Max value diff: {max_value_diff:.8f} < {threshold}")
    else:
        print(f"❌ FAILED - RMA with alpha=0 is NOT equivalent to original PPO")
        print(f"   Max action diff: {max_action_diff:.8f} >= {threshold}")
        print(f"   Max value diff: {max_value_diff:.8f} >= {threshold}")
    print("=" * 80)
    
    return max_action_diff, max_value_diff

if __name__ == '__main__':
    test_rma_equivalence()
