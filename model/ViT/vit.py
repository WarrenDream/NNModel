import torch
import torch.nn as nn

class Indentity(nn.Module):
    def __init__(self):
        super(Indentity,self).__init__()
    def forward(self,x):
        return x
class Mlp(nn.Module):
    def __init__(self, embed_dim, mlp_ratio=4.0, dropout=0.):
        super(Mlp,self).__init__()
        self.fc1 = nn.Linear(embed_dim, int(embed_dim*mlp_ratio))
        self.fc2 = nn.Linear(int(embed_dim*mlp_ratio),embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        


if__name__=="__main__":
    main()
