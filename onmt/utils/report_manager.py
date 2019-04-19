""" Report manager utility """
from __future__ import print_function
import time
from datetime import datetime
import codecs
from nltk import word_tokenize
from nltk.translate.bleu_score import *
import onmt
from itertools import izip
from statistics import mean
from onmt.utils.sari import SARIsent
from onmt.utils.logging import logger

smooth = SmoothingFunction()

def BLEU_file(refs,preds="pred.txt"):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [preds, refs]]
    references = []
    hypothese = []
    for pred, ref in izip(*files):
        references.append([word_tokenize(r) for r in ref.split('\n')])
        hypothese.append(word_tokenize(pred))
    for fis in files:
        fis.close()
    # Smoothing method 3: NIST geometric sequence smoothing
    return corpus_bleu(references, hypothese, smoothing_function=smooth.method3)
def SARI_file(source, refs,preds="pred.txt"):
    files = [codecs.open(fis, "r", 'utf-8') for fis in [source, preds, refs]]
    scores = []
    for src, pred, ref in izip(*files):
        references = [r for r in ref.split('\n')]
        scores.append(SARIsent(src, pred, references))
    for fis in files:
        fis.close()
    return mean(scores)

def build_report_manager(opt):
    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        tensorboard_log_dir = opt.tensorboard_log_dir

        if not opt.train_from:
            tensorboard_log_dir += datetime.now().strftime("/%b-%d_%H-%M-%S")

        writer = SummaryWriter(tensorboard_log_dir,
                               comment="Unmt")
    else:
        writer = None

    report_mgr = ReportMgr(opt.report_every, start_time=-1,
                           tensorboard_writer=writer,source=opt.valid_src,dest=opt.valid_dest)
    return report_mgr


class ReportMgrBase(object):
    """
    Report Manager Base class
    Inherited classes should override:
        * `_report_training`
        * `_report_step`
    """

    def __init__(self, report_every, start_time=-1.):
        """
        Args:
            report_every(int): Report status every this many sentences
            start_time(float): manually set report start time. Negative values
                means that you will need to set it later or use `start()`
        """
        self.report_every = report_every
        self.progress_step = 0
        self.start_time = start_time

    def start(self):
        self.start_time = time.time()

    def log(self, *args, **kwargs):
        logger.info(*args, **kwargs)

    def report_training(self, step, num_steps, learning_rate,
                        report_stats):
        """
        This is the user-defined batch-level traing progress
        report function.

        Args:
            step(int): current step count.
            num_steps(int): total number of batches.
            learning_rate(float): current learning rate.
            report_stats(Statistics): old Statistics instance.
        Returns:
            report_stats(Statistics): updated Statistics instance.
        """
        if self.start_time < 0:
            raise ValueError("""ReportMgr needs to be started
                                (set 'start_time' or use 'start()'""")

        if step % self.report_every == 0:

            self._report_training(
                step, num_steps, learning_rate, report_stats)
            self.progress_step += 1
            return onmt.utils.Statistics()
        else:
            return report_stats

    def _report_training(self, *args, **kwargs):
        """ To be overridden """
        raise NotImplementedError()

    def report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        Report stats of a step

        Args:
            train_stats(Statistics): training stats
            valid_stats(Statistics): validation stats
            lr(float): current learning rate
        """
        self._report_step(
            lr, step, train_stats=train_stats, valid_stats=valid_stats)

    def _report_step(self, *args, **kwargs):
        raise NotImplementedError()


class ReportMgr(ReportMgrBase):
    def __init__(self, report_every, start_time=-1., tensorboard_writer=None,source=None,dest=None):
        """
        A report manager that writes statistics on standard output as well as
        (optionally) TensorBoard

        Args:
            report_every(int): Report status every this many sentences
            tensorboard_writer(:obj:`tensorboard.SummaryWriter`):
                The TensorBoard Summary writer to use or None
        """
        super(ReportMgr, self).__init__(report_every, start_time)
        self.tensorboard_writer = tensorboard_writer
        self.source=source
        self.dest=dest

    def maybe_log_tensorboard(self, stats, prefix, learning_rate, step):
        if self.tensorboard_writer is not None:
            stats.log_tensorboard(
                prefix, self.tensorboard_writer, learning_rate, step)

    def _report_training(self, step, num_steps, learning_rate,
                         report_stats):
        """
        See base class method `ReportMgrBase.report_training`.
        """
        report_stats.output(step, num_steps,
                            learning_rate, self.start_time)

        # Log the progress using the number of batches on the x-axis.
        self.maybe_log_tensorboard(report_stats,
                                   "progress",
                                   learning_rate,
                                   self.progress_step)
        report_stats = onmt.utils.Statistics()

        return report_stats

    def _report_step(self, lr, step, train_stats=None, valid_stats=None):
        """
        See base class method `ReportMgrBase.report_step`.
        """
        if train_stats is not None:
            self.log('Train perplexity: %g' % train_stats.ppl())
            self.log('Train accuracy: %g' % train_stats.accuracy())

            self.maybe_log_tensorboard(train_stats,
                                       "train",
                                       lr,
                                       step)

        if valid_stats is not None:
            self.log('Validation perplexity: %g' % valid_stats.ppl())
            self.log('Validation accuracy: %g' % valid_stats.accuracy())
            self.log('Validation bleu: %g' % BLEU_file(self.dest))
            self.log('Validation Sari: %g' % SARI_file(self.source,self.dest))

            self.maybe_log_tensorboard(valid_stats,
                                       "valid",
                                       lr,
                                       step)
