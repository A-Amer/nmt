"""Module defining decoders."""
from onmt.decoders.decoder import DecoderBase, InputFeedRNNDecoder
from onmt.decoders.transformer import TransformerDecoder



str2dec = {"rnn": InputFeedRNNDecoder, "ifrnn": InputFeedRNNDecoder,
            "transformer": TransformerDecoder}

__all__ = ["DecoderBase", "TransformerDecoder",
           "InputFeedRNNDecoder", "str2dec"]
