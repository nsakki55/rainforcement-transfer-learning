import numpy as np
import torch

prob = np.array([0.3, 0.4, 0.3])
log = np.array([np.log(p) for p in prob])

print(sum(-prob * log))


prob = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
log = np.array([np.log(p) for p in prob])

print(sum(-prob * log))

l_list = []
for _ in range(5):
    l = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9])
    print(l)
    l_list.append(l)

print(torch.stack(l_list).transpose(0, 1))

a = torch.randn(5, 2)
print(a)
print(a.mean(0))

def get_random_policy(policy_length, lr_count):
    policy_nums = [i for i in range(lr_count)]
    
    return np.random.choice(policy_nums, policy_length)

policy = get_random_policy(17, 4)
print(policy)