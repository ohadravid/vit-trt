
import torch
import torch.nn as nn
import torch.nn.functional as F

class ExplicitAttention(nn.Module):
    """
    The explicit, original version of the Attention layer from the VideoMAEv2 codebase.
    """

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class AttentionUsingScaledDotProduct(nn.Module):
    """
    An alternative implementation of the Attention layer using `F.scaled_dot_product_attention`, which is ~50% faster,
    but doesn't compile correctly when using TensorRT v10.
    """
    
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        x = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.attn_drop.p if self.training else 0.0,
            scale=self.scale,
        )
        
        x = x.transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class AttentionUsingMHAForward(nn.Module):
    """
    An alternative implementation of the Attention layer using `F.multi_head_attention_forward`, which has the same performance as the original implementation,
    but compiles correctly when using TensorRT v10.
    """
    
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 attn_head_dim=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(all_head_dim))
            self.v_bias = nn.Parameter(torch.zeros(all_head_dim))
        else:
            self.q_bias = None
            self.v_bias = None

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv_bias = None
        if self.q_bias is not None:
            qkv_bias = torch.cat(
                (self.q_bias,
                 torch.zeros_like(self.v_bias,
                                  requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # MHA expects [sequence, batch, embed_dim].
        x_t = x.transpose(0, 1)  # => [N, B, C]

        # Originally, VideoMAEv2 define `forward` using `attn = (q * self.scale) @ k.transpose(-2, -1))`.
        # (See https://github.com/OpenGVLab/VideoMAEv2/blob/master/models/modeling_finetune.py#L172)
        # We changed this to `x = F.scaled_dot_product_attention(q, k, v, ...)`, which has optimized CUDA kernels.
        # However, after upgrading to TensorRT v10, the resulting ONNX compiled incorrectly to TRT engine, resulting in random outputs.
        # This is fixed by replacing the entire forward function with `F.multi_head_attention_forward`.
        attn_out, _ = F.multi_head_attention_forward(
            x_t,
            x_t,
            x_t,
            embed_dim_to_check=C,
            num_heads=self.num_heads,
            # Since use_separate_proj_weight=False (default), then according to the docs:
            # "in_proj_weight will be used, which is a combination of q_proj_weight, k_proj_weight, v_proj_weight."
            in_proj_weight=self.qkv.weight,
            in_proj_bias=qkv_bias,
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=self.attn_drop.p,
            out_proj_weight=self.proj.weight,
            out_proj_bias=self.proj.bias,
            training=self.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=None,
        )

        # Transpose back to [B, N, C].
        x = attn_out.transpose(0, 1)

        return x


class AttentionUsingMHALayer(nn.MultiheadAttention):
    """
    An alternative implementation of the Attention layer using `nn.MultiheadAttention`, which has the higher performance of the scaled dot-product attention,
    and compiles correctly when using TensorRT v10.
    """

    _version = 2

    def __init__(self,
                dim,
                num_heads=8,
                qkv_bias=False,
                qk_scale=None,
                attn_drop=0.,
                proj_drop=0.,
                attn_head_dim=None):
        assert qk_scale is None or qk_scale is True, f"qk_scale is not supported in this class, got {qk_scale}"
        assert attn_head_dim is None, f"attn_head_dim is not supported in this class, got {attn_head_dim}"
        assert proj_drop == attn_drop, f"proj_drop must be equal to attn_drop, got {proj_drop} and {attn_drop}"

        super().__init__(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, bias=qkv_bias, add_bias_kv=False, batch_first=True)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        version = local_metadata.get("version", None)

        if version is None or version < 2:
            # The old layer uses `q_bias` and `v_bias` to construct `qkv_bias`.
            q_bias = state_dict.pop(f"{prefix}q_bias")
            v_bias = state_dict.pop(f"{prefix}v_bias")
            if q_bias is not None:
                qkv_bias = torch.cat(
                    (q_bias, torch.zeros_like(v_bias, requires_grad=False), v_bias)
                )
                state_dict[f"{prefix}in_proj_bias"] = qkv_bias

            key_mapping = {
                "qkv.weight": "in_proj_weight",
                "proj.weight": "out_proj.weight",
                "proj.bias": "out_proj.bias",
            }

            # The rest of the keys only require a rename.
            for from_key, to_key in key_mapping.items():
                old_key = f"{prefix}{from_key}"
                new_key = f"{prefix}{to_key}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
    
    def forward(self, x):
        # On macOS, need_weights=True is actually faster.
        need_weights = x.device.type == "mps"
        attn_output, attn_output_weights = super().forward(query=x, key=x, value=x, need_weights=need_weights)
        return attn_output
