import torch
import torch.nn as nn
from torch import flatten

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
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, dropout=0.):
        super(PatchEmbedding,self).__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size = patch_size, stride=patch_size,bias=False)
        self.patch_embed.weight=nn.Parameter(nn.init.constant_(self.patch_embed.weight,1.0))
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x):
        # x: [1,1,28,28] 
        x = self.patch_embed(x)  # x: [n, embed_dim, h, w]
        x = x.flatten(2) # [n, embed_dim, h*w] 
        x = x.permute(0, 2, 1) #[n, h*w, embed_dim]
        x = self.dropout(x)
        return x
class Encoder(nn.Module):
    def __init__(sefl, embed_dim):
        super().__init__()
        self.attn = Indentity()
        self.attn_norm = nn.LayerNorm(embed_dim)
        self.mlp = Mlp(embed_dim)
        self.mlp_norm = nn.LayerNorm(embed_dim)
    
    def forward(self,x):
        h = x
        x = self.attn_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.mlp_norm(x)
        x = self.mlp(x)
        x = x + h
class ViT(nn.Module):
    def __init__(self):
        super(ViT,self).__init__()
        self.patch_embed = PatchEmbedding(224, 7, 3, 16)

        layer_list = [Encoder(16) for i in range(5)]
        self.encoders = nn.ModuleList(layer_list)
        self.head = nn.Linear(16,10) # num_classes
        self.avgpool = nn.AdaptiveAvgpool1d(1)

    def forward(self, x):
        x = self.patch_embed(x)
        for encoder in self.encoders:
            x = encoder(x)

        # [ n, h*w, c]
        x = x.permute(0,2,1)
        x = self.avgpool(x) # [n, c, 1]
        x = x.flatten(1) #[n, c]
        x = self.head(x)
        return x
    
def main():
    t = torch.randn([4, 3, 244, 244])
    model = ViT()
    out = model(t)

    print("out shape=", out.shape)




if __name__=='__main__':
    main()

