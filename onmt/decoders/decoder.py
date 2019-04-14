import torch
import torch.nn as nn

from onmt.models.stacked_rnn import StackedLSTM, StackedGRU
from onmt.modules import  GlobalAttention
from onmt.utils.rnn_factory import rnn_factory

from onmt.utils.misc import aeq


class DecoderBase(nn.Module):
    """Abstract class for decoders.

    Args:
        attentional (bool): The decoder returns non-empty attention.
    """

    def __init__(self, attentional=True):
        super(DecoderBase, self).__init__()
        self.attentional = attentional

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor.

        Subclasses should override this method.
        """

        raise NotImplementedError


class RNNDecoderBase(DecoderBase):
    """Base recurrent attention-based decoder class.

    Specifies the interface used by different decoder types
    and required by :class:`~onmt.models.NMTModel`.


    .. mermaid::

       graph BT
          A[Input]
          subgraph RNN
             C[Pos 1]
             D[Pos 2]
             E[Pos N]
          end
          G[Decoder State]
          H[Decoder State]
          I[Outputs]
          F[memory_bank]
          A--emb-->C
          A--emb-->D
          A--emb-->E
          H-->C
          C-- attn --- F
          D-- attn --- F
          E-- attn --- F
          C-->I
          D-->I
          E-->I
          E-->G
          F---I

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional_encoder (bool) : use with a bidirectional encoder
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       attn_type (str) : see :class:`~onmt.modules.GlobalAttention`
       attn_func (str) : see :class:`~onmt.modules.GlobalAttention`
       coverage_attn (str): see :class:`~onmt.modules.GlobalAttention`
       context_gate (str): see :class:`~onmt.modules.ContextGate`
       copy_attn (bool): setup a separate copy attention mechanism
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use

    """

    def __init__(self, rnn_type, bidirectional_encoder, num_layers,
                 hidden_size, attn_type="general", attn_func="softmax",
                 dropout=0.0, embeddings=None):
        super(RNNDecoderBase, self).__init__(
            attentional=attn_type != "none" and attn_type is not None)

        self.bidirectional_encoder = bidirectional_encoder
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.dropout = nn.Dropout(dropout)

        # Decoder state
        self.state = {}

        # Build the RNN.
        self.rnn = self._build_rnn(rnn_type,
                                   input_size=self._input_size,
                                   hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   dropout=dropout)


        self.attn = GlobalAttention(
            hidden_size,
            attn_type=attn_type, attn_func=attn_func
        )


    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.dec_layers,
            opt.dec_rnn_size,
            opt.global_attention,
            opt.global_attention_function,
            opt.dropout,
            embeddings)

    def init_state(self, src, memory_bank, encoder_final,enc_reshape):
        """Initialize decoder state with last state of the encoder."""
        def _fix_enc_hidden(hidden):
            # The encoder hidden is  (layers*directions) x batch x dim.
            # We need to convert it to layers x batch x (directions*dim).
            if self.bidirectional_encoder and enc_reshape:
                hidden = torch.cat([hidden[0:hidden.size(0):2],
                                    hidden[1:hidden.size(0):2]], 2)
            return hidden

        if isinstance(encoder_final, tuple):  # LSTM
            self.state["hidden"] = tuple(_fix_enc_hidden(enc_hid)
                                         for enc_hid in encoder_final)
        else:  # GRU
            self.state["hidden"] = (_fix_enc_hidden(encoder_final), )

        # Init the input feed.
        batch_size = self.state["hidden"][0].size(1)
        print(self.state["hidden"][0].size())
        h_size = (batch_size, self.hidden_size)
        self.state["input_feed"] = \
            self.state["hidden"][0].data.new(*h_size).zero_().unsqueeze(0)

    def map_state(self, fn):
        self.state["hidden"] = tuple(fn(h, 1) for h in self.state["hidden"])
        self.state["input_feed"] = fn(self.state["input_feed"], 1)

    def detach_state(self):
        self.state["hidden"] = tuple(h.detach() for h in self.state["hidden"])
        self.state["input_feed"] = self.state["input_feed"].detach()

    def forward(self, tgt, memory_bank, memory_lengths=None, step=None):
        """
        Args:
            tgt (LongTensor): sequences of padded tokens
                 ``(tgt_len, batch, nfeats)``.
            memory_bank (FloatTensor): vectors from the encoder
                 ``(src_len, batch, hidden)``.
            memory_lengths (LongTensor): the padded source lengths
                ``(batch,)``.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * dec_outs: output from the decoder (after attn)
              ``(tgt_len, batch, hidden)``.
            * attns: distribution over src at each tgt
              ``(tgt_len, batch, src_len)``.
        """

        dec_state, dec_outs, attns = self._run_forward_pass(
            tgt, memory_bank, memory_lengths=memory_lengths)

        # Update the state with the result.
        if not isinstance(dec_state, tuple):
            dec_state = (dec_state,)
        self.state["hidden"] = dec_state
        self.state["input_feed"] = dec_outs[-1].unsqueeze(0)
        

        # Concatenates sequence of tensors along a new dimension.
        # NOTE: v0.3 to 0.4: dec_outs / attns[*] may not be list
        #       (in particular in case of SRU) it was not raising error in 0.3
        #       since stack(Variable) was allowed.
        #       In 0.4, SRU returns a tensor that shouldn't be stacke
        if type(dec_outs) == list:
            dec_outs = torch.stack(dec_outs)

            for k in attns:
                if type(attns[k]) == list:
                    attns[k] = torch.stack(attns[k])
        return dec_outs, attns



class InputFeedRNNDecoder(RNNDecoderBase):
    """Input feeding based decoder.

    See :class:`~onmt.decoders.decoder.RNNDecoderBase` for options.

    Based around the input feeding approach from
    "Effective Approaches to Attention-based Neural Machine Translation"
    :cite:`Luong2015`


    .. mermaid::

       graph BT
          A[Input n-1]
          AB[Input n]
          subgraph RNN
            E[Pos n-1]
            F[Pos n]
            E --> F
          end
          G[Encoder]
          H[memory_bank n-1]
          A --> E
          AB --> F
          E --> H
          G --> H
    """

    def _run_forward_pass(self, tgt, memory_bank, memory_lengths=None):
        """
        See StdRNNDecoder._run_forward_pass() for description
        of arguments and return values.
        """
        # Additional args check.
        input_feed = self.state["input_feed"].squeeze(0)
        input_feed_batch, _ = input_feed.size()
        _, tgt_batch, _ = tgt.size()
        aeq(tgt_batch, input_feed_batch)
        # END Additional args check.

        dec_outs = []
        attns = {}
        if self.attn is not None:
            attns["std"] = []
    

        emb = self.embeddings(tgt)
        assert emb.dim() == 3  # len x batch x embedding_dim

        dec_state = self.state["hidden"]
        
        # Input feed concatenates hidden state with
        # input at every time step.
        for emb_t in emb.split(1):
            decoder_input = torch.cat([emb_t.squeeze(0), input_feed], 1)
            rnn_output, dec_state = self.rnn(decoder_input, dec_state)
            print(memory_bank.transpose(0, 1).size())
            if self.attentional:
                decoder_output, p_attn = self.attn(
                    rnn_output,
                    memory_bank.transpose(0, 1),
                    memory_lengths=memory_lengths)
                attns["std"].append(p_attn)
            else:
                decoder_output = rnn_output
           
            decoder_output = self.dropout(decoder_output)
            input_feed = decoder_output

            dec_outs += [decoder_output]

        return dec_state, dec_outs, attns

    def _build_rnn(self, rnn_type, input_size,
                   hidden_size, num_layers, dropout):
        stacked_cell = StackedLSTM if rnn_type == "LSTM" else StackedGRU
        return stacked_cell(num_layers, input_size, hidden_size, dropout)

    @property
    def _input_size(self):
        """Using input feed by concatenating input with attention vectors."""
        return self.embeddings.embedding_size + self.hidden_size
