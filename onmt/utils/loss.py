"""
This includes: LossComputeBase and the standard NMTLossCompute, and
               sharded loss compute stuff.
"""
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import onmt

def _build_target_tokens(vocab, pred,eos_token):
        tokens = []
        for tok in pred:
            tokens.append(vocab.itos[tok])
            if tokens[-1] == eos_token:
                tokens = tokens[:-1]
                break
        return tokens[:-1]
def build_loss_compute(model, tgt_field, opt, train=True):
    """
    Returns a LossCompute subclass which wraps around an nn.Module subclass
    (such as nn.NLLLoss) which defines the loss criterion.
    """
    device = torch.device("cuda" if onmt.utils.misc.use_gpu(opt) else "cpu")

    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    if opt.label_smoothing > 0 and train:
        criterion = LabelSmoothingLoss(
            opt.label_smoothing, len(tgt_field.vocab), ignore_index=padding_idx
        )
    else:
       criterion = nn.NLLLoss(ignore_index=padding_idx, reduction='sum')
        
    loss_gen = model.generator
    compute = NMTLossCompute(criterion, loss_gen,vocab=tgt_field.vocab,eos_token=tgt_field.eos_token)
    compute.to(device)

    return compute


class LossComputeBase(nn.Module):
    """
    Class for managing efficient loss computation. Handles
    sharding next step predictions and accumulating multiple
    loss computations

    Users can implement their own loss computation strategy by making
    subclass of this one.  Users need to implement the _compute_loss()
    and make_shard_state() methods.

    Args:
        generator (:obj:`nn.Module`) :
             module that maps the output of the decoder to a
             distribution over the target vocabulary.
        tgt_vocab (:obj:`Vocab`) :
             torchtext vocab object representing the target output
        normalzation (str): normalize by "sents" or "tokens"
    """

    def __init__(self, criterion, generator):
        super(LossComputeBase, self).__init__()
        self.criterion = criterion
        self.generator = generator

    @property
    def padding_idx(self):
        return self.criterion.ignore_index

    def _make_shard_state(self, batch, output, range_, attns=None):
        """
        Make shard state dictionary for shards() to return iterable
        shards for efficient loss computation. Subclass must define
        this method to match its own _compute_loss() interface.
        Args:
            batch: the current batch.
            output: the predict output from the model.
            range_: the range of examples for computing, the whole
                    batch
            attns: the attns dictionary returned from the model.
        """
        return NotImplementedError

    def _compute_loss(self, batch, output, target, **kwargs):
        """
        Compute the loss. Subclass must define this method.

        Args:

            batch: the current batch.
            output: the predict output from the model.
            target: the validate target to compare output with.
            **kwargs(optional): additional info for computing loss.
        """
        return NotImplementedError

    def __call__(self,
                 batch,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=0, valid=False):
        """Compute the forward loss, possibly in shards in which case this
        method also runs the backward pass and returns ``None`` as the loss
        value.

        Args:
          batch (batch) : batch of labeled examples
          output (:obj:`FloatTensor`) :
              output of decoder model `[tgt_len x batch x hidden]`
          attns (dict) : dictionary of attention distributions
              `[tgt_len x batch x src_len]`
          normalization: Optional normalization factor.
          shard_size (int) : maximum number of examples in a shard

        Returns:
            A tuple with the loss and a :obj:`onmt.utils.Statistics` instance.
        """
        trunc_range = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            loss, stats,preds = self._compute_loss(batch, **shard_state,valid=valid,prediction_type="greedy")
            return [loss.div(float(normalization))], stats,preds
        batch_stats = onmt.utils.Statistics()

        preds=[]
        for shard in shards(shard_state, shard_size):
            loss, stats,pred = self._compute_loss(batch, **shard,valid=valid,prediction_type="greedy")
            preds.append(pred)
            loss.div(float(normalization)).backward()
            batch_stats.update(stats)
        return None, batch_stats,preds

    def get_losses(self,
                 batch,
                 output,
                 attns,
                 normalization=1.0,
                 shard_size=0, valid=False
                   ,prediction_type="greedy"):

        trunc_range = (0, batch.tgt.size(0))
        shard_state = self._make_shard_state(batch, output, trunc_range, attns)
        if shard_size == 0:
            loss, stats,preds = self._compute_loss(batch, **shard_state,valid=valid,prediction_type=prediction_type)
            return [loss.div(float(normalization))], stats,preds
        batch_stats = onmt.utils.Statistics()
        losses=[]
        preds=[]
        for shard in shards_no_backprop(shard_state, shard_size):
            loss, stats,pred = self._compute_loss(batch, **shard,valid=valid,prediction_type=prediction_type)
            preds.append(pred)
            losses.append(loss.div(float(normalization)))
            batch_stats.update(stats)
        return losses, batch_stats,preds

    def _stats(self, loss, scores, target):
        """
        Args:
            loss (:obj:`FloatTensor`): the loss computed by the loss criterion.
            scores (:obj:`FloatTensor`): a score for each possible output
            target (:obj:`FloatTensor`): true targets

        Returns:
            :obj:`onmt.utils.Statistics` : statistics for this batch.
        """
        pred = scores.max(1)[1]
        
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target).masked_select(non_padding).sum().item()
        num_non_padding = non_padding.sum().item()
        return onmt.utils.Statistics(loss.item(), num_non_padding, num_correct)

    def _bottle(self, _v):
        return _v.view(-1, _v.size(2))

    def _unbottle(self, _v, batch_size):
        return _v.view(-1, batch_size, _v.size(1))


class NMTLossCompute(LossComputeBase):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, criterion, generator,vocab=None,eos_token=2, normalization="sents"):
        super(NMTLossCompute, self).__init__(criterion, generator)
        self.tgt_vocab=vocab
        self.eos_token=eos_token

    def _make_shard_state(self, batch, output, range_, attns=None):
        return {
            "output": output,
            "target": batch.tgt[range_[0] + 1: range_[1], :, 0],
        }

    def _compute_loss(self, batch, output, target, valid=False,prediction_type="greedy"):
        bottled_output = self._bottle(output)

        scores = self.generator(bottled_output)
        gtruth = target.view(-1)
        if prediction_type == "greedy":
            _, pred = scores.max(1)
            pred = torch.autograd.Variable(pred, requires_grad=False)
            loss = self.criterion(scores,  gtruth)
            loss_data = loss.data.clone()
        elif prediction_type == "sample":
            logits=scores
            dist = torch.distributions.Multinomial(
                logits=logits, total_count=1)
            topk_ids = torch.argmax(dist.sample(), dim=1, keepdim=True).view(-1)
            pred = topk_ids
            loss = self.criterion(scores,  pred)
            loss_data = loss.sum().data
        else:
            raise ValueError("Incorrect prediction_type %s" % prediction_type)
        stats = self._stats(loss_data, scores, gtruth)
        pred=pred.view(target.size(1),target.size(0))
        if valid:
          pred_file=open("pred.txt",'a')
          pred = scores.max(1)[1]
          for i in range(output.size()[1]):
            out_tokens=_build_target_tokens(self.tgt_vocab, pred,self.eos_token)
            sentence=' '.join(word for word in out_tokens)
            pred_file.write(sentence+'\n')
          pred_file.close()
        return loss, stats,pred
class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence = 1.0 - label_smoothing

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """
        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='sum')

def filter_shard_state(state, shard_size=None):
    for k, v in state.items():
        if shard_size is None:
            yield k, v

        if v is not None:
            v_split = []
            if isinstance(v, torch.Tensor):
                for v_chunk in torch.split(v, shard_size):
                    v_chunk = v_chunk.data.clone()
                    v_chunk.requires_grad = v.requires_grad
                    v_split.append(v_chunk)
            yield k, (v, v_split)


def shards(state, shard_size, eval_only=False):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.
        eval_only: If True, only yield the state, nothing else.
              Otherwise, yield shards.

    Yields:
        Each yielded shard is a dict.

    Side effect:
        After the last shard, this function does back-propagation.
    """
    if eval_only:
        yield filter_shard_state(state)
    else:
        # non_none: the subdict of the state dictionary where the values
        # are not None.
        non_none = dict(filter_shard_state(state, shard_size))

        # Now, the iteration:
        # state is a dictionary of sequences of tensor-like but we
        # want a sequence of dictionaries of tensors.
        # First, unzip the dictionary into a sequence of keys and a
        # sequence of tensor-like sequences.
        keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                             for k, (_, v_split) in non_none.items()))

        # Now, yield a dictionary for each shard. The keys are always
        # the same. values is a sequence of length #keys where each
        # element is a sequence of length #shards. We want to iterate
        # over the shards, not over the keys: therefore, the values need
        # to be re-zipped by shard and then each shard can be paired
        # with the keys.
        for shard_tensors in zip(*values):
            yield dict(zip(keys, shard_tensors))

        # Assumed backprop'd
        variables = []
        for k, (v, v_split) in non_none.items():
            if isinstance(v, torch.Tensor) and state[k].requires_grad:
                variables.extend(zip(torch.split(state[k], shard_size),
                                     [v_chunk.grad for v_chunk in v_split]))
        inputs, grads = zip(*variables)
        torch.autograd.backward(inputs, grads)

def shards_no_backprop(state, shard_size):
    """
    Args:
        state: A dictionary which corresponds to the output of
               *LossCompute._make_shard_state(). The values for
               those keys are Tensor-like or None.
        shard_size: The maximum size of the shards yielded by the model.


    Yields:
        Each yielded shard is a dict.

    """

    # non_none: the subdict of the state dictionary where the values
    # are not None.
    non_none = dict(filter_shard_state(state, shard_size))


    keys, values = zip(*((k, [v_chunk for v_chunk in v_split])
                         for k, (_, v_split) in non_none.items()))


    for shard_tensors in zip(*values):
        yield dict(zip(keys, shard_tensors))
