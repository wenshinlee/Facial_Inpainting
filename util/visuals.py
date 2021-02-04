import os
from util import utils
from torch.utils.tensorboard import SummaryWriter
import socket
from datetime import datetime


class Visuals(object):
    def __init__(self, checkpoints_dir, model):
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join(checkpoints_dir,
                               'runs', model + '_' + current_time + socket.gethostname())
        if not os.path.exists(log_dir):
            utils.mkdirs(log_dir)

        self.writer = SummaryWriter(log_dir=log_dir)

    def add_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)

    def add_image(self, tag, value, step):
        self.writer.add_image(tag, value, step)
