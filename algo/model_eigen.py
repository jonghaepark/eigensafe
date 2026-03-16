import torch
import torch.nn as nn
import torch.nn.functional as F

import os


# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

class Psi(nn.Module):
    def __init__(self, num_inputs, args):
        super(Psi, self).__init__()
        
        self.lr = args.lr
        self.hidden_dim = args.hidden_size
        self.device = torch.device("cuda")
        
        # Psi architecture
        self.linear1 = nn.Linear(num_inputs, self.hidden_dim)
        self.linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.linear3 = nn.Linear(self.hidden_dim, 1)

        # Lambda
        self.eigenvalue = nn.Parameter(torch.tensor(0.9, dtype=torch.float32), requires_grad=True)
        
        self.apply(weights_init_)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

    def forward(self, x1):
        
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1

    def forward_np(self, x1):
        
        x1 = torch.from_numpy(x1).float().to(self.device)
        x1 = F.relu(self.linear1(x1))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)

        return x1.detach().cpu().numpy()

    def update_parameters(self, agent, memory, args):
        # Sample batch (safe == 1, unsafe == 0)
        state_batch, safety_batch, action_batch, reward_batch, next_state_batch, next_safety_batch, mask_batch = memory.sample(batch_size=args.batch_size)
        
        # Convert to tensor
        state_batch = torch.from_numpy(state_batch).to(self.device)
        action_batch = torch.from_numpy(action_batch).to(self.device)
        next_state_batch = torch.from_numpy(next_state_batch).to(self.device)
        
        safety_batch = torch.from_numpy(safety_batch).to(self.device).unsqueeze(1)
        next_safety_batch = torch.from_numpy(next_safety_batch).to(self.device).unsqueeze(1)

        # Sample next action
        next_action_batch, _, _ = agent.policy.sample(next_state_batch)

        # Compute psi(s,a) and psi(s',a')
        sa = torch.cat([state_batch, action_batch], dim=1)
        next_sa = torch.cat([next_state_batch, next_action_batch], dim=1)

        psi = self.forward(sa) * safety_batch
        psi_next = self.forward(next_sa) * safety_batch * next_safety_batch

        # Eigenfunction loss: (ψ' - λψ)^2
        eigen_loss = torch.mean((psi_next.detach() - self.eigenvalue * psi) ** 2)

        # Normalization loss
        norm_loss = (1 - torch.max(psi))**2

        # Positivity loss
        pos_loss = torch.mean(torch.relu(-psi))

        # Combine (you can weight them if needed)
        loss = eigen_loss + norm_loss + pos_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), torch.mean(psi).item(), self.eigenvalue
    
    # Save model parameters
    def save_checkpoint(self, args, suffix="", ckpt_path=None):
        ckpt_dir = os.path.join("results", args.exp_name, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        if ckpt_path is None:
            ckpt_path = os.path.join(ckpt_dir, "psi_checkpoint_{}_{}".format(args.env_name, suffix))
        print('Saving model to {}'.format(ckpt_path))
        torch.save({'model_state_dict': self.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}, ckpt_path)

    def load_checkpoint(self, ckpt_path, evaluate=False):
        print('Loading model from {}'.format(ckpt_path))
        if ckpt_path is not None:
            checkpoint = torch.load(ckpt_path)
            self.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if evaluate:
                self.eval()
            else:
                self.train()

