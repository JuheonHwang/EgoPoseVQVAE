"""Helpers shared by the entry-point scripts (train*.py / test*.py).

Nothing here imports torch at module level: ``init_gpu`` has to run before
torch is imported for ``CUDA_VISIBLE_DEVICES`` to take effect.
"""
import argparse
import logging
import os

__all__ = ['init_gpu', 'get_device', 'setup_logging', 'override_output']


def init_gpu():
    """Set CUDA_VISIBLE_DEVICES from ``--gpu`` in sys.argv (if given).

    Call it at the top of a script, before ``import torch``.
    """

    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument('--gpu')
    args, _ = parser.parse_known_args()
    if args.gpu is not None:
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def get_device():
    """First visible GPU, or the CPU when CUDA is not available."""

    import torch
    return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def setup_logging(log_file):
    """Log plain messages to ``log_file`` (overwritten); returns the logger."""

    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    logging.basicConfig(filename=log_file, format='%(message)s',
                        level=logging.INFO, filemode='w')
    return logging.getLogger('train_logger')


def override_output(config, output_dir):
    """Redirect checkpoints and log under ``output_dir`` (no-op when None),
    keeping the layout of the config: ``<output_dir>/ckpts/<ckpt_dir name>/``
    and ``<output_dir>/<log file name>``."""

    if output_dir is not None:
        out = config.output
        out.ckpt_dir = os.path.join(output_dir, 'ckpts', os.path.basename(os.path.normpath(out.ckpt_dir)))
        out.log_file = os.path.join(output_dir, os.path.basename(out.log_file))
