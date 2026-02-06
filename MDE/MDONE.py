import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """Simple fully-connected MLP."""
    def __init__(self, in_dim, hidden_dims, out_dim, activation=nn.GELU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(activation())
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: [..., in_dim]
        return self.net(x)


class MultiBranchDeepONet(nn.Module):
    """
    DeepONet with 3 branch nets and 1 trunk net.
    
    Inputs:
      u1: [B, b1]
      u2: [B, b2]
      u3: [B, b3]
      x:  [B, t1, d_x]   # sampling points (coordinates)
    
    Output:
      y:  [B, t1]
    """
    
    def __init__(
        self,
        b1_dim: int,
        b2_dim: int,
        b3_dim: int,
        trunk_in_dim: int,
        width: int = 64,
        branch_hidden=(128, 128),
        trunk_hidden=(128, 128),
    ):
        super().__init__()
        # Three branch networks, each mapping its own sensor values -> latent width
        self.branch1 = MLP(b1_dim, branch_hidden, width)
        self.branch2 = MLP(b2_dim, branch_hidden, width)
        self.branch3 = MLP(b3_dim, branch_hidden, width)
        self.branch4 = MLP(b3_dim, branch_hidden, width)

        # Trunk net acts pointwise on coordinates
        self.trunk = MLP(trunk_in_dim, trunk_hidden, width)

    def forward(self, u1, u2, u3,u4, x):
        """
        u1: [B, b1]
        u2: [B, b2]
        u3: [B, b3]
        x:  [B, t1, d_x]
        """
        B, t1, d_x = x.shape

        # ----- Branch side -----
        b1 = self.branch1(u1)  # [B, width]
        b2 = self.branch2(u2)  # [B, width]
        b3 = self.branch3(u3)  # [B, width]
        b4 = self.branch3(u4)  # [B, width]

        # Elementwise product across the 3 branches -> one combined branch vector
        # shape: [B, width]
        branch = b1 * b2 * b3 * b4

        # ----- Trunk side -----
        # Apply the trunk net pointwise to each coordinate x_i
        # Flatten batch and t1 dims, then reshape back
        x_flat = x.reshape(B * t1, d_x)      # [B*t1, d_x]
        t_flat = self.trunk(x_flat)          # [B*t1, width]
        trunk_feats = t_flat.view(B, t1, -1) # [B, t1, width]

        # ----- DeepONet combination -----
        # vanilla DeepONet: elementwise product in latent dim and sum over latent dim
        # branch:      [B, width]      -> [B, 1, width]
        # trunk_feats: [B, t1, width]
        y = (branch.unsqueeze(1) * trunk_feats).sum(dim=-1)  # [B, t1]

        return y
if __name__ == "__main__":
    B = 4
    b1, b2, b3 = 16, 32, 8
    t1, d_x = 50, 1

    model = MultiBranchDeepONet(
        b1_dim=b1,
        b2_dim=b2,
        b3_dim=b3,
        trunk_in_dim=d_x,
        width=64,
    )

    u1 = torch.randn(B, b1)
    u2 = torch.randn(B, b2)
    u3 = torch.randn(B, b3)
    x  = torch.linspace(0, 1, t1).view(1, t1, 1).expand(B, -1, -1)  # [B, t1, 1]

    y = model(u1, u2, u3, x)  # y: [B, t1]
    print(y.shape)            # torch.Size([4, 50])
