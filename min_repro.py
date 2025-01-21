import torch
import torch.nn as nn
import torch.nn.functional as F
import io
import tensorrt as trt
import torch_tensorrt


class AttentionUsingScaledDotProduct(nn.Module):
    """
    An alternative implementation of the Attention layer using `F.scaled_dot_product_attention`, which is ~50% faster,
    but doesn't compile correctly when using TensorRT v10.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
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
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
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


class ExplicitAttention(nn.Module):
    """
    The explicit, original version of the Attention layer from the VideoMAEv2 codebase.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
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
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class AttentionUsingMHAForward(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
    ):
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
                (
                    self.q_bias,
                    torch.zeros_like(self.v_bias, requires_grad=False),
                    self.v_bias,
                )
            )
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        # MHA expects [sequence, batch, embed_dim].
        x_t = x.transpose(0, 1)  # => [N, B, C]

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


def onnx_to_trt(onnx_bytes: bytes) -> bytes:
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)

    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)

    parser.parse(onnx_bytes)

    config = builder.create_builder_config()
    config.builder_optimization_level = 0

    engine = builder.build_serialized_network(network, config)

    trt_bytes = io.BytesIO()
    trt_bytes.write(engine)

    return trt_bytes.getvalue()


def build_trt_module(model, x):
    onnx_bytes = io.BytesIO()

    torch.onnx.export(
        model,
        (x,),
        onnx_bytes,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["x"],
        output_names=["y"],
    )

    trt_engine = onnx_to_trt(onnx_bytes.getvalue())

    model = torch_tensorrt.runtime.PythonTorchTensorRTModule(
        trt_engine,
        input_binding_names=[
            "x",
        ],
        output_binding_names=[
            "y",
        ],
    )

    return model


@torch.inference_mode()
def main():
    torch.manual_seed(0)

    EMB_DIM = 384
    x = torch.rand((6, 1568, EMB_DIM))

    explicit_attention = ExplicitAttention(EMB_DIM)
    sdpa = AttentionUsingScaledDotProduct(EMB_DIM)
    mha_fwd = AttentionUsingMHAForward(EMB_DIM)

    # Use the same params for all.
    sdpa.load_state_dict(explicit_attention.state_dict())
    mha_fwd.load_state_dict(explicit_attention.state_dict())

    sdpa_torch_y = sdpa(x)
    explicit_attention_torch_y = explicit_attention(x)
    mha_fwd_torch_y = mha_fwd(x)

    print(
        "Torch: [explicit<->sdpa] Is allclose?",
        sdpa_torch_y.allclose(explicit_attention_torch_y, atol=0.0001),
    )
    print(
        "Torch: [explicit<->mha_fwd] Is allclose?",
        mha_fwd_torch_y.allclose(explicit_attention_torch_y, atol=0.0001),
    )
    print(
        "Torch: [explicit<->sdpa] Total difference:",
        (sdpa_torch_y - explicit_attention_torch_y).abs().sum(),
    )
    print(
        "Torch: [explicit<->mha_fwd] Total difference:",
        (mha_fwd_torch_y - explicit_attention_torch_y).abs().sum(),
    )
    assert sdpa_torch_y.allclose(explicit_attention_torch_y, atol=0.0001), "Precheck"
    assert mha_fwd_torch_y.allclose(explicit_attention_torch_y, atol=0.0001), "Precheck"

    explicit_attention_trt = build_trt_module(explicit_attention, x)
    sdpa_trt_model = build_trt_module(sdpa, x)
    mha_fwd_trt_model = build_trt_module(mha_fwd, x)

    explicit_attention_y = explicit_attention_trt(x.cuda())
    sdpa_y = sdpa_trt_model(x.cuda())
    mha_fwd_y = mha_fwd_trt_model(x.cuda())

    print(
        "TRT: [explicit<->sdpa] Is allclose?",
        sdpa_y.allclose(explicit_attention_y, atol=0.0001),
    )
    print(
        "TRT: [explicit<->sdpa] Total difference:",
        (sdpa_y - explicit_attention_y).abs().sum(),
    )

    print(
        "TRT: [explicit<->mha_fwd] Is allclose?",
        mha_fwd_y.allclose(explicit_attention_y, atol=0.0001),
    )
    print(
        "TRT: [explicit<->mha_fwd] Total difference:",
        (mha_fwd_y - explicit_attention_y).abs().sum(),
    )

    print("TRT: Explicit Attention:", explicit_attention_y[0, 0, :32])
    print("TRT: Scaled Dot Product Attention:", sdpa_y[0, 0, :32])
    print("TRT: MHA Forward:", mha_fwd_y[0, 0, :32])


if __name__ == "__main__":
    main()
