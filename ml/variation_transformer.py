import torch 
from torch import nn




class SeqVarTransformer(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 num_heads: int
                 ):
        super(SeqVarTransformer).__init__()
        self.in_dim = in_dim 
        self.in_ln = nn.LayerNorm(self.in_dim)
        self.attn_1 = torch.nn.MultiheadAttention(self.in_dim, num_heads)

        self.ln2 = nn.LayerNorm(self.in_dim)
        self.attn_2 = torch.nn.MultiheadAttention(self.in_dim, num_heads)

        self.ln3 = nn.LayerNorm(self.in_dim)
        self.mean_mlp = nn.Linear(self.in_dim, self.in_dim)
        self.var_mlp = nn.Linear(self.in_dim, self.in_dim)
        self.proj = nn.Linear(2*self.in_dim, self.in_dim)
        self.ln4 = nn.LayerNorm(self.in_dim)

        



    def forward(self, response, mask=None):
        x = self.in_ln(response)
        assert mask.shape == x.shape
        attn_x, _ = self.attn_1(x, x, x, attn_mask=mask)

        out_1 = attn_x + response

        x_2 = self.ln2(out_1)
        x_2, _ = self.attn_2(x_2, x_2, x_2)
        out_2 = x_2 + out_1

        x_3 = self.ln3(out_2)

        x_mean = self.mean_mlp(x_3)
        x_var = self.var_mlp(x_3)
        x_var = torch.exp(x_var)
        sample = torch.randn(x_3.shape) 
        z = sample * x_var + x_mean

        # concatnate
        out =  self.proj(torch.cat((x_3, z), dim=1))
        final_out = out + out_2
        out_x = self.ln4(final_out)
        return out_x












        













def main():
    bs = 100
    input_dim = 256 # seq len
    response = torch.randn(bs, input_dim)
    mask = torch.triu(response)

    model = SeqVarTransformer()
    out = model(response, mask=mask)






if __name__ == "__main__":
    main()