# from audiocraft.modules import StreamingTransformer
import torch
from torch.nn import functional as F
import sys
import math

state = torch.load("./assets/small/weight/state_dict.bin")
weights = state["best_state"]
seq = torch.ones((4, 1024))
print(seq)
x = seq

_x = torch.ones(1, 4, 5).to(torch.float32).type(torch.LongTensor)
for i in range(4):
    _x[:,i] = (i+1) * _x[:, i]
print("_x", x, x.shape)

print("_x[:, 0]", _x[:, 0], _x[:, 0].shape)
print("_x[:, 1]", _x[:, 1], _x[:, 1].shape)
print("_x[:, 1]", _x[:, 2], _x[:, 2].shape)
print("_x[:, 1]", _x[:, 3], _x[:, 3].shape)
emb = weights["emb.0.weight"].to(torch.float32)
res = F.embedding(_x[:, 0], emb)

emb = weights["emb.1.weight"].to(torch.float32)
res = res + F.embedding(_x[:, 1], emb)

emb = weights["emb.2.weight"].to(torch.float32)
res = res + F.embedding(_x[:, 2], emb)

emb = weights["emb.3.weight"].to(torch.float32)
res = res + F.embedding(_x[:, 3], emb)
print("res", res, res.shape)
sys.exit(0)

n_w = weights["transformer.layers.0.norm1.weight"].to(torch.float32)
n_b = weights["transformer.layers.0.norm1.bias"].to(torch.float32)

print("n_w shape", n_w.shape)
print("input shape", seq.shape)
x = F.layer_norm(seq, n_w.shape, n_w, n_b)
print("after norm", x, x.shape)

# now do self attn
embed_dim = 1024
num_heads = 16
num_layers = 1024
dim_ff = 4096
kv_repeat = 1

per_head_dim = embed_dim // num_heads
kv_heads = num_heads // kv_repeat
start = embed_dim
end = start + per_head_dim * kv_heads

in_proj_weight = weights["transformer.layers.0.self_attn.in_proj_weight"].to(
    torch.float32
)
projected = F.linear(x, in_proj_weight)
print("in_proj_weight", in_proj_weight, in_proj_weight.shape)
print("after linear", projected, projected.shape)


q = projected[:, :embed_dim]
k = projected[:, start:end]
v = projected[:, end:]

print("self_attn q", q, q.shape)
print("self_attn k", k, k.shape)
print("self_attn v", v, v.shape)

out_proj_weight = weights["transformer.layers.0.self_attn.out_proj.weight"].to(
    torch.float32
)

dp_self_attn = F.scaled_dot_product_attention(q, k, v)
print("dp attention before linear", dp_self_attn, dp_self_attn.shape)
dp_self_attn = F.linear(dp_self_attn, out_proj_weight)
print("dp attention after linear", dp_self_attn, dp_self_attn.shape)

(expected_attn, _) = F.multi_head_attention_forward(
    x,
    x,
    x,
    embed_dim,
    num_heads,
    in_proj_weight,
    None,
    None,
    None,
    False,
    0.0,
    out_proj_weight,
    None,
)

print("expected attention", expected_attn, expected_attn.shape)


x = x + expected_attn
print("after all self_attn (x + expected_attn)", x, x.shape)

# cross attn normalization
crn_w = weights["transformer.layers.0.norm_cross.weight"].to(torch.float32)
crn_b = weights["transformer.layers.0.norm_cross.bias"].to(torch.float32)

x = F.layer_norm(x, crn_w.shape, crn_w, crn_b)
print("after cross layernorm", x, x.shape)


cross_in_proj_weight = weights[
    "transformer.layers.0.cross_attention.in_proj_weight"
].to(torch.float32)
cross_out_proj_weight = weights[
    "transformer.layers.0.cross_attention.out_proj.weight"
].to(torch.float32)

cross_attn_src = torch.ones([4, 1024])  # placeholder just like the other stuffs

dim = cross_in_proj_weight.shape[0] // 3
print("dim", dim)


print("cross_attn_src", cross_attn_src, cross_attn_src.shape)
print("c_in_proj_w", cross_in_proj_weight, cross_in_proj_weight.shape)
print("c_in_proj_w", cross_in_proj_weight[:dim])
print("c_in_proj_w", cross_in_proj_weight[dim : 2 * dim], cross_in_proj_weight[dim : 2 * dim].shape)
print("x shape", x.shape)

q = F.linear(x, cross_in_proj_weight[:dim])
print("q", q, q.shape)
k = F.linear(cross_attn_src, cross_in_proj_weight[dim : 2 * dim])
print("k", k, k.shape)
v = F.linear(cross_attn_src, cross_in_proj_weight[2 * dim :])
print("v", v, v.shape)

cross_dp_attention = F.scaled_dot_product_attention(q, k, v)
print("cross_dp_attention before linear", cross_dp_attention, cross_dp_attention.shape)
cross_dp_attention = F.linear(cross_dp_attention, cross_out_proj_weight)
print("cross_dp_attention after linear", cross_dp_attention, cross_dp_attention.shape)

cross_attn_expected, _ = F.multi_head_attention_forward(
    x,
    cross_attn_src,
    cross_attn_src,
    embed_dim,
    num_heads,
    cross_in_proj_weight,
    None,
    None,
    None,
    False,
    0.0,
    cross_out_proj_weight,
    None,
)

print("cross_attn_expected", cross_attn_expected, cross_attn_expected.shape)

x = x + cross_attn_expected
print("after all cross_attn (x + cross_attn_expected)", x, x.shape)

n_w2 = weights["transformer.layers.0.norm2.weight"].to(torch.float32)
print("n_w2", n_w2, n_w2.shape)
n_b2 = weights["transformer.layers.0.norm2.bias"].to(torch.float32)
print("n_b2", n_b2, n_b2.shape)

x = F.layer_norm(x, n_w2.shape, n_w2, n_b2)
print("x after second layernorm", x, x.shape)

linear1_w = weights["transformer.layers.0.linear1.weight"].to(torch.float32)
linear2_w = weights["transformer.layers.0.linear2.weight"].to(torch.float32)

x_p = F.linear(x, linear1_w)
x_p = F.gelu(x_p)
x_p = F.linear(x_p, linear2_w)
print("x_p after linear layers", x_p, x_p.shape)

x = x + x_p
print("x after linear layers", x, x.shape)
