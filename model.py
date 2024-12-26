from tinygrad import Tensor, nn

def Projection(d_i, d_o):
  return (Tensor.scaled_uniform(d_i, d_o), Tensor.zeros(d_o))

class TransformerBlock:
  def __init__(self,
               embed_dim=8,
               num_heads=2,
               ff_dim=8,
               prenorm=False,
               act=lambda x: x.relu(),
               dropout=0.1):
    assert embed_dim % num_heads == 0, "embed dim not divisible by head dim"
    self.num_heads = num_heads
    self.head_size = embed_dim // num_heads
    self.prenorm, self.act, self.dropout = prenorm, act, dropout

    self.Q = Projection(embed_dim, embed_dim)
    self.K = Projection(embed_dim, embed_dim)
    self.V = Projection(embed_dim, embed_dim)

    self.O = Projection(embed_dim, embed_dim)

    self.ff1 = Projection(embed_dim, ff_dim)
    self.ff2 = Projection(ff_dim, embed_dim)

    self.ln1 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))
    self.ln2 = (Tensor.ones(embed_dim), Tensor.zeros(embed_dim))

  def attn(self, x):
    # x: (bs, time, embed_dim) -> (bs, time, embed_dim)
    Qx, Kx, Vx = [x.linear(*y).reshape(shape=(x.shape[0], -1, self.num_heads, self.head_size)).transpose(1,2) for y in [self.Q, self.K, self.V]]
    attention = Tensor.scaled_dot_product_attention(Qx, Kx, Vx).transpose(1,2)
    return attention.reshape(shape=(x.shape[0], -1, self.num_heads * self.head_size)).linear(*self.O)

  def __call__(self, x):
    if self.prenorm:
      x = x + self.attn(x.layernorm().linear(*self.ln1)).dropout(self.dropout)
      x = x + self.act(x.layernorm().linear(*self.ln2).linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
    else:
      x = x + self.attn(x).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln1)
      x = x + self.act(x.linear(*self.ff1)).linear(*self.ff2).dropout(self.dropout)
      x = x.layernorm().linear(*self.ln2)
    return x

class UniversalTransformer:
  def __init__(self,
               n_steps=8,
               embed_dim=8,
               num_heads=2,
               ff_dim=8,
               prenorm=False,
               act=lambda x: x.relu(),
               dropout=0.1):
    self.transformer_block = TransformerBlock(embed_dim,
                                              num_heads, 
                                              ff_dim,
                                              prenorm,
                                              act,
                                              dropout)
    self.n_steps = n_steps
    self.step_embed = Tensor.scaled_uniform(n_steps, embed_dim)

    # GRU weights for update gate, reset gate, and candidate
    self.Wz = Tensor.scaled_uniform(embed_dim, embed_dim)
    self.Uz = Tensor.scaled_uniform(embed_dim, embed_dim)
    self.Wr = Tensor.scaled_uniform(embed_dim, embed_dim)
    self.Ur = Tensor.scaled_uniform(embed_dim, embed_dim)
    self.Wh = Tensor.scaled_uniform(embed_dim, embed_dim)
    self.Uh = Tensor.scaled_uniform(embed_dim, embed_dim)

  def transition_gru(self, h, x):
    # Update gate
    z = (x.linear(self.Wz) + h.linear(self.Uz)).sigmoid()
    # Reset gate
    r = (x.linear(self.Wr) + h.linear(self.Ur)).sigmoid()
    # Candidate hidden state
    h_tilde = (x.linear(self.Wh) + (r * h).linear(self.Uh)).tanh()
    # Final hidden state
    return (1 - z) * h + z * h_tilde

  def __call__(self, x):
    h = x
    for step in range(self.n_steps):
      step_embedding = self.step_embed[step]
      h_t = self.transformer_block(x + step_embedding)
      h = self.transition_gru(h, h_t)
    return h



if __name__ == "__main__":
  bs = 2
  s_len = 1024
  e_dim = 8

  x = Tensor.randn(bs, s_len, e_dim)

  print("=== testing transformer block ===")
  transformer = TransformerBlock()
  t_out = transformer(x)

  n_steps = 6
  
  print("=== testing universal transformer ===")
  ut = UniversalTransformer(embed_dim=e_dim, n_steps=n_steps)
  output = ut(x)
