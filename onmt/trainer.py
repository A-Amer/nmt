
import torch
import traceback
import onmt.utils
from onmt.utils.logging import logger
from onmt.utils.sari import SARIsent
import numpy as np

class Scorer:
    def __init__(self, rouge_weight, sari_weight,eos_idx):
        import rouge as R
        self.rouge = R.Rouge(stats=["f"], metrics=[
            "rouge-1", "rouge-2", "rouge-l"])
        self.r_weight = rouge_weight
        self.s_weight = sari_weight
        self.eos_idx=eos_idx

    def score_rouge(self, hyps, refs):
        scores = self.rouge.get_scores(hyps, refs)
        # NOTE: here we use score = r1 * r2 * rl
        #       I'm not sure how relevant it is
        metric_weight = {"rouge-1": 0, "rouge-2": 0, "rouge-l": 1}

        scores = [sum([seq[metric]['f'] * metric_weight[metric]
                       for metric in seq.keys()])
                  for seq in scores]
        return np.array(scores)

    def score_sari(self, hyps, refs, srcs):
        scores = []
        for i in range(len(refs)):
            scores.append(SARIsent(srcs[i], hyps[i], [refs[i]]))
        #consider adding reverse sari [beta*SARIsent(srcs[i], refs[i], [hyps[i]])+(1-beta)*scores] where beta=0.1
        return np.array(scores)

    def tens2sen(self, t):
        sentences = []
        for s in t:
            sentence = []
            for wt in s:
                if wt in [self.eos_idx]:
                    break
                sentence += [str(wt)]
            if len(sentence) == 0:
                # NOTE just a trick not to score empty sentence
                #      this has not consequence
                sentence = ["0", "0", "0"]
            sentences += [" ".join(sentence)]
        return sentences

    def score(self, sample_pred, greedy_pred, tgt, src):
        """
            sample_pred: LongTensor [bs x len]
            greedy_pred: LongTensor [bs x len]
            tgt: LongTensor [bs x len]
        """

        s_hyps = self.tens2sen(sample_pred)
        g_hyps = self.tens2sen(greedy_pred)
        refs = self.tens2sen(tgt)
        srcs = self.tens2sen(src)
        sample_scores = self.score_rouge(s_hyps, refs) * self.r_weight + self.score_sari(s_hyps, refs,
                                                                                         srcs) * self.s_weight
        greedy_scores = self.score_rouge(g_hyps, refs) * self.r_weight + self.score_sari(g_hyps, refs,
                                                                                         srcs) * self.s_weight

        ts = torch.Tensor(sample_scores)
        gs = torch.Tensor(greedy_scores)

        return (gs - ts)

def build_trainer(opt, device_id, model, fields, optim, model_saver=None):
    """
    Simplify `Trainer` creation based on user `opt`s*

    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """

    tgt_field = dict(fields)["tgt"].base_field
    train_loss = onmt.utils.loss.build_loss_compute(model, tgt_field, opt)
    valid_loss = onmt.utils.loss.build_loss_compute(
        model, tgt_field, opt, train=False)
    shard_size = opt.max_generator_batches if opt.model_dtype == 'fp32' else 0
    norm_method = opt.normalization
    accum_count = opt.accum_count
    accum_steps = opt.accum_steps
    n_gpu = 1
    if device_id >= 0:
        gpu_rank = 0
    else:
        gpu_rank = 0
        n_gpu = 0
    gpu_verbose_level = opt.gpu_verbose_level

    earlystopper = onmt.utils.EarlyStopping(
        opt.early_stopping, scorers=onmt.utils.scorers_from_opts(opt)) \
        if opt.early_stopping > 0 else None
    rl=opt.rl
    rl_only=opt.rl_only
    gamma=opt.gamma
    if rl:
        scorer=Scorer(opt.rouge_weight,opt.sari_weight,tgt_field.vocab.stoi[tgt_field.eos_token])
    else:
        scorer=None
    report_manager = onmt.utils.build_report_manager(opt)
    trainer = onmt.Trainer(model, train_loss, valid_loss, optim,
                           shard_size, norm_method,
                           accum_count, accum_steps,
                           n_gpu, gpu_rank,
                           gpu_verbose_level, report_manager,
                           model_saver=model_saver if gpu_rank == 0 else None,
                           model_dtype=opt.model_dtype,
                           earlystopper=earlystopper,
                           rl=rl,rl_only=rl_only,gamma=gamma,scorer=scorer)
    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text]
            norm_method(string): normalization methods: [sents|tokens]
            accum_count(list): accumulate gradients this many times.
            accum_steps(list): steps for accum gradients changes.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
            rl(bool):Whether to use reinforcement learning
            rl_only(bool):Wheter to use reinforcement loss only or apply equation (gamma*rl_loss)+(gamma-1)*ml_loss
            gamma(float):gama used in loss calculation
    """

    def __init__(self, model, train_loss, valid_loss, optim,
                 shard_size=32,
                 norm_method="sents", accum_count=[1],
                 accum_steps=[0],
                 n_gpu=1, gpu_rank=1,
                 gpu_verbose_level=0, report_manager=None, model_saver=None,
                model_dtype='fp32',
                 earlystopper=None,
                 rl=False,rl_only=True,gamma=0.9,scorer=None):
        # Basic attributes.
        self.model = model
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.optim = optim
        self.shard_size = shard_size
        self.norm_method = norm_method
        self.accum_count_l = accum_count
        self.accum_count = accum_count[0]
        self.accum_steps = accum_steps
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.gpu_verbose_level = gpu_verbose_level
        self.report_manager = report_manager
        self.model_saver = model_saver
        self.model_dtype = model_dtype
        self.earlystopper = earlystopper
        self.rl=rl
        self.rl_only=rl_only
        self.gamma=gamma
        self.scorer=scorer
        for i in range(len(self.accum_count_l)):
            assert self.accum_count_l[i] > 0

        # Set model in training mode.
        self.model.train()

    def _accum_count(self, step):
        for i in range(len(self.accum_steps)):
            if step > self.accum_steps[i]:
                _accum = self.accum_count_l[i]
        return _accum

    def _accum_batches(self, iterator):
        batches = []
        normalization = 0
        self.accum_count = self._accum_count(self.optim.training_step)
        for batch in iterator:
            batches.append(batch)
            if self.norm_method == "tokens":
                num_tokens = batch.tgt[1:, :, 0].ne(
                    self.train_loss.padding_idx).sum()
                normalization += num_tokens.item()
            else:
                normalization += batch.batch_size
            if len(batches) == self.accum_count:
                yield batches, normalization
                self.accum_count = self._accum_count(self.optim.training_step)
                batches = []
                normalization = 0
        if batches:
            yield batches, normalization



    def train(self,
              train_iter,
              train_steps,
              save_checkpoint_steps=5000,
              valid_iter=None,
              valid_steps=10000):
        """
        The main training loop by iterating over `train_iter` and possibly
        running validation on `valid_iter`.

        Args:
            train_iter: A generator that returns the next training batch.
            train_steps: Run training for this many iterations.
            save_checkpoint_steps: Save a checkpoint every this many
              iterations.
            valid_iter: A generator that returns the next validation batch.
            valid_steps: Run evaluation every this many iterations.

        Returns:
            The gathered statistics.
        """
        if valid_iter is None:
            logger.info('Start training loop without validation...')
        else:
            logger.info('Start training loop and validate every %d steps...',
                        valid_steps)

        total_stats = onmt.utils.Statistics()
        report_stats = onmt.utils.Statistics()
        self._start_report_manager(start_time=total_stats.start_time)


        for i, (batches, normalization) in enumerate(
                self._accum_batches(train_iter)):
            step = self.optim.training_step

            if self.gpu_verbose_level > 1:
                logger.info("GpuRank %d: index: %d", self.gpu_rank, i)
            if self.gpu_verbose_level > 0:
                logger.info("GpuRank %d: reduce_counter: %d \
                            n_minibatch %d"
                            % (self.gpu_rank, i + 1, len(batches)))

            self._gradient_accumulation(
                batches, normalization, total_stats,
                report_stats)


            report_stats = self._maybe_report_training(
                step, train_steps,
                self.optim.learning_rate(),
                report_stats)

            if valid_iter is not None and step % valid_steps == 0:
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: validate step %d'
                                % (self.gpu_rank, step))
                valid_stats = self.validate( valid_iter)
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: gather valid stat \
                                step %d' % (self.gpu_rank, step))
                if self.gpu_verbose_level > 0:
                    logger.info('GpuRank %d: report stat step %d'
                                % (self.gpu_rank, step))
                self._report_step(self.optim.learning_rate(),
                                  step, valid_stats=valid_stats)
                # Run patience mechanism
                if self.earlystopper is not None:
                    self.earlystopper(valid_stats, step)
                    # If the patience has reached the limit, stop training
                    if self.earlystopper.has_stopped():
                        break

            if (self.model_saver is not None
                and (save_checkpoint_steps != 0
                     and step % save_checkpoint_steps == 0)):
                self.model_saver.save(step)

            if train_steps > 0 and step >= train_steps:
                break

        if self.model_saver is not None:
            self.model_saver.save(step)
        return total_stats

    def validate(self, valid_iter):
        """ Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        valid_model = self.model

        # Set model in validating mode.
        valid_model.eval()
        pred_file=open("pred.txt", "w")
        pred_file.close()
        with torch.no_grad():
            stats = onmt.utils.Statistics()

            for batch in valid_iter:
                src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                                   else (batch.src, None)
                tgt = batch.tgt

                # F-prop through the model.
                outputs, attns = valid_model(src, tgt, src_lengths)

                # Compute loss.
                _, batch_stats,_ = self.valid_loss(batch, outputs, attns,valid=True)

                # Update statistics.
                stats.update(batch_stats)
        
        valid_model.train()

        return stats

    def _gradient_accumulation(self, true_batches, normalization, total_stats,
                               report_stats):


        for k, batch in enumerate(true_batches):
            target_size = batch.tgt.size(0)
            # Truncated BPTT: reminder not compatible with accum > 1
            src, src_lengths = batch.src if isinstance(batch.src, tuple) \
                else (batch.src, None)
            if src_lengths is not None:
                report_stats.n_src_words += src_lengths.sum().item()

            # 1. Create truncated target.
            tgt = batch.tgt

            # 2. F-prop all but generator.
            self.optim.zero_grad()
            outputs, attns = self.model(src, tgt, src_lengths, bptt=False)

            # 3. Compute loss.
            try:

                if self.rl:
                    loss, batch_stats, preds = self.train_loss.get_losses(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size)
                    loss_sample, _, preds_sample = self.train_loss.get_losses(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size,prediction_type="sample")
                    metric = self.scorer.score(preds_sample[0], preds[0], tgt[1:].squeeze(2).t(),src.squeeze(2).t())
                    loss_sample=loss_sample[0]
                    loss=loss[0]
                    if self.n_gpu>0:
                        metric = metric.cuda()
                        loss_sample=loss_sample.cuda()
                        loss=loss.cuda()
                    rl_loss = (loss_sample * metric).sum()
                    if self.rl_only:
                        loss = rl_loss
                    else:
                        loss = (self.gamma * rl_loss) - ((1 - self.gamma * loss))
                else:
                    loss, batch_stats, preds = self.train_loss(
                        batch,
                        outputs,
                        attns,
                        normalization=normalization,
                        shard_size=self.shard_size)

                if loss is not None:
                    self.optim.backward(loss)
                    
                total_stats.update(batch_stats)
                report_stats.update(batch_stats)

            except Exception:
                traceback.print_exc()
                logger.info("At step %d, we removed a batch - accum %d",
                            self.optim.training_step, k)

            # 4. Update the parameters and statistics.

            self.optim.step()
            # If truncated, don't backprop fully.
            # TO CHECK
            # if dec_state is not None:
            #    dec_state.detach()
            if self.model.decoder.state is not None:
                self.model.decoder.detach_state()



    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time


    def _maybe_report_training(self, step, num_steps, learning_rate,
                               report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats)

    def _report_step(self, learning_rate, step, train_stats=None,
                     valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats,
                valid_stats=valid_stats)
