import sys
import torch

if len(sys.argv) != 2:
    print("Usage: python check_checkpoint.py <checkpoint_path>")
    sys.exit(1)

path = sys.argv[1]
print(f"Loading checkpoint: {path}")
checkpoint = torch.load(path, map_location='cpu')

model_state_dict = checkpoint['model_state_dict']
print(f"\nTotal parameters: {len(model_state_dict)}")
print("\nParameter list:")
for i, (key, val) in enumerate(model_state_dict.items()):
    print(f"  {i+1:2d}. {key:40s} shape={list(val.shape)}")

print("\nChecking actor structure:")
actor_keys = [k for k in model_state_dict.keys() if k.startswith('actor.')]
for k in sorted(actor_keys):
    print(f"  {k:40s} shape={list(model_state_dict[k].shape)}")

print("\nChecking critic structure:")
critic_keys = [k for k in model_state_dict.keys() if k.startswith('critic.')]
for k in sorted(critic_keys):
    print(f"  {k:40s} shape={list(model_state_dict[k].shape)}")

if 'std' in model_state_dict:
    print(f"\nstd shape: {model_state_dict['std'].shape}")

if 'history_encoder' in str(list(model_state_dict.keys())):
    print("\n⚠️ This checkpoint contains RMA modules!")
else:
    print("\n✅ This is a pure PPO checkpoint (no RMA)")