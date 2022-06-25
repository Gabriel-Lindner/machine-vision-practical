import re
import torch
import sys
class ProgressBar(object):
    DEFAULT = 'Progress: %(bar)s %(percent)3d%%'
    FULL = '%(bar)s %(current)d/%(total)d (%(percent)3d%%) %(remaining)d to go'

    def __init__(self, total, width=40, fmt=DEFAULT, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
            r'\g<name>%dd' % len(str(total)), fmt)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        print('\r' + self.fmt % args, file=self.output, end='')

    def done(self):
        self.current = self.total
        self()
        print('', file=self.output)


def save(path, model, optimizer, **kwargs):
    model_state = None
    optimizer_state = None
    if model is not None:
        model_state = model.state_dict()
    if optimizer is not None:
        optimizer_state = optimizer.state_dict()
    torch.save(
        dict(model_state=model_state,
             optimizer_state=optimizer_state,
             **kwargs),
        path
    )


def resume(path, model, optimizer):
    if torch.cuda.is_available():
        snapshot = torch.load(path)
    else:
        snapshot = torch.load(path, map_location="cpu")
    print("Loaded snapshot from", path)

    model_state = snapshot.pop('model_state', snapshot)
    optimizer_state = snapshot.pop('optimizer_state', None)

    if model is not None and model_state is not None:
        print("loading model...")
        model.load_state_dict(model_state)

    if optimizer is not None and optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
    return snapshot
