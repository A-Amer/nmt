"""Define RNN-based encoders."""
import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory


class ResidualEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type,num_layers,
                 hidden_size, dropout=0.0,gnmt=False,embeddings=None):
        super(ResidualEncoder, self).__init__()
        assert embeddings is not None
        assert num_layers>1
        self.embeddings = embeddings
        self.gnmt=gnmt
        self.num_layers=num_layers
        self.enc_reshape=True
        self.layers=nn.ModuleList()
        self.dropout=nn.Dropout(dropout)
        bottom_layers=1
        if gnmt:
            bi_rnn, self.no_pack_padded_seq = \
                rnn_factory(rnn_type,
                            input_size=embeddings.embedding_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=True)
            rnn, _ = \
                rnn_factory(rnn_type,
                            input_size=hidden_size*2,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=False)
            self.layers.append(bi_rnn)
            self.layers.append(rnn)
            bottom_layers=2
        else:
            rnn, self.no_pack_padded_seq=\
                rnn_factory(rnn_type,
                            input_size=embeddings.embedding_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=False)
            self.layers.append(rnn)

        for i in range(num_layers-bottom_layers):
            rnn, _= \
                rnn_factory(rnn_type,
                            input_size=hidden_size,
                            hidden_size=hidden_size,
                            num_layers=1,
                            bidirectional=False)
            self.layers.append(rnn)


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout,opt.gnmt,
            embeddings)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        # s_len, batch, emb_dim = emb.size()

        packed_emb = emb
        if lengths is not None and not self.no_pack_padded_seq:
            # Lengths data is wrapped inside a Tensor.
            lengths_list = lengths.view(-1).tolist()
            packed_emb = pack(emb, lengths_list)

        memory_bank, encoder_final = self.layers[0](packed_emb)

        if lengths is not None and not self.no_pack_padded_seq:
            memory_bank = unpack(memory_bank)[0]
        
        bottom_layers = 1
        if self.gnmt:
            memory_bank = self.dropout(memory_bank)
            memory_bank, encoder_final= self.layers[1](memory_bank)
            bottom_layers=2
        for i in range(bottom_layers,self.num_layers):
            residual = memory_bank
            memory_bank = self.dropout(memory_bank)
            memory_bank, enc_final = self.layers[i](memory_bank)
            encoder_final0  = torch.cat((encoder_final[0] , enc_final[0]), 0)
            encoder_final1  = torch.cat((encoder_final[1] , enc_final[1]), 0)
            encoder_final=(encoder_final0,encoder_final1)
            memory_bank = memory_bank+ residual
        print(encoder_final[0].size())

        return encoder_final, memory_bank, lengths
