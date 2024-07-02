import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        main_path = nn.Sequential(nn.Linear(dim, hidden_dim),
                                  nn.ReLU(), 
                                  nn.Linear(hidden_dim, dim))
        
        self.res = nn.Sequential(nn.Residual(main_path),
                                 nn.ReLU())

    def forward(self, x):
        return self.res(x)
    


class ResidualMLP(nn.Module):
    def __init__(self, dim, hidden_dim=128, num_blocks=3, num_classes=10):
        super().__init__()
        self.resnet = nn.Sequential(nn.Linear(dim, hidden_dim),
                                    nn.ReLU(),
                                    *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2) for _ in range(num_blocks)],
                                    nn.Linear(hidden_dim, num_classes))
    
    def forward(self, x):
        return self.resnet(x)