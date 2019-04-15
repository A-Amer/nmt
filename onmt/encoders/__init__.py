"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder
from onmt.encoders.residual_encoder import ResidualEncoder



str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder,"res":ResidualEncoder,
           "transformer": TransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder","ResidualEncoder", "str2enc"]
