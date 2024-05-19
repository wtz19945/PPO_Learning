import torch
from torch.utils.tensorboard import SummaryWriter

# Create a SummaryWriter for logging
writer = SummaryWriter('logs')

# Dummy training loop
for i in range(100):
    # Example: log scalar value (e.g., loss)
    writer.add_scalar('loss', i**2, i)

# Close the SummaryWriter
writer.close()