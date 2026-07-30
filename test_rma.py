import sys
sys.path.insert(0, '/home/gjt/legged_gym')
sys.path.insert(0, '/home/gjt/isaacgym/python/examples/rsl_rl')

import torch
from legged_gym.algorithms.rma_actor_critic import RMAActorCritic

print("=" * 60)
print("RMA Module Test")
print("=" * 60)

print("\n1. Testing RMAActorCritic initialization...")
net = RMAActorCritic(
    num_actor_obs=48,
    num_critic_obs=48,
    num_actions=12,
    history_len=10,
    latent_dim=32
)
print("   ✓ Network created")

print("\n2. Testing forward pass...")
num_envs = 64
obs = torch.randn(num_envs, 48)
history = torch.randn(num_envs, 10, 48)

latent = net.compute_latent(history)
print(f"   ✓ Latent shape: {latent.shape}")
assert latent.shape == (num_envs, 32), f"Expected (64, 32), got {latent.shape}"

action = net.act(obs, obs_history=history)
print(f"   ✓ Action shape: {action.shape}")
assert action.shape == (num_envs, 12), f"Expected (64, 12), got {action.shape}"

value = net.evaluate(obs, obs_history=history)
print(f"   ✓ Value shape: {value.shape}")
assert value.shape == (num_envs, 1), f"Expected (64, 1), got {value.shape}"

print("\n3. Testing act_inference (for deployment)...")
action_inf = net.act_inference(obs, obs_history=history)
print(f"   ✓ Inference action shape: {action_inf.shape}")

print("\n4. Testing CUDA support...")
if torch.cuda.is_available():
    net.cuda()
    obs_cuda = obs.cuda()
    history_cuda = history.cuda()
    action_cuda = net.act(obs_cuda, obs_history=history_cuda)
    print(f"   ✓ CUDA action shape: {action_cuda.shape}")
    print(f"   ✓ CUDA action device: {action_cuda.device}")
else:
    print("   - CUDA not available, skipping")

print("\n5. Testing critic with obs_history=None (PPO update compatibility)...")
net2 = RMAActorCritic(
    num_actor_obs=48,
    num_critic_obs=48,
    num_actions=12,
    history_len=10,
    latent_dim=32
)
value_no_hist = net2.evaluate(obs, obs_history=None)
print(f"   ✓ Value without history shape: {value_no_hist.shape}")
assert value_no_hist.shape == (num_envs, 1), f"Expected (64, 1), got {value_no_hist.shape}"

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
print("\nSummary:")
print(f"  num_actor_obs: {net.num_actor_obs}")
print(f"  num_critic_obs: {net.num_critic_obs}")
print(f"  num_actions: {net.num_actions}")
print(f"  history_len: {net.history_len}")
print(f"  latent_dim: {net.latent_dim}")
print(f"  Actor input dim: {net.num_actor_obs + net.latent_dim}")
print(f"  Critic input dim: {net.num_critic_obs + net.latent_dim}")
