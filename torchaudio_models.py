from torchaudio.models import Conformer

conformer = Conformer(
    input_dim=80,
    num_heads=4,
    ffn_dim=128,
    num_layers=4,
    depthwise_conv_kernel_size=31,
)
lengths = torch.randint(1, 400, (10,))  # (batch,)
input = torch.rand(10, int(lengths.max()), input_dim)  # (batch, num_frames, input_dim)
output = conformer(input, lengths)
