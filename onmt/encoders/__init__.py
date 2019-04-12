"""Module defining encoders."""
from onmt.encoders.encoder import EncoderBase
from onmt.encoders.transformer import TransformerEncoder
from onmt.encoders.rnn_encoder import RNNEncoder



str2enc = {"rnn": RNNEncoder, "brnn": RNNEncoder,
           "transformer": TransformerEncoder}

__all__ = ["EncoderBase", "TransformerEncoder", "RNNEncoder", "str2enc"]
