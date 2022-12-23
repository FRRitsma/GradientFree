# %%
import torch


x = torch.Tensor([1])
x.requires_grad = True
loss = torch.abs(x)

optimizer = torch.optim.Adam([x], lr=float(1e-1))

back = loss.backward()

optimizer.step()