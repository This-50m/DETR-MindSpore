from collections import deque


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.queue = deque(maxlen=window_size)
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        if self.count != 0:
            self.queue.clear()
            self.count = 0
        self.avg = 0
        self.sum = 0

    def update(self, val, weight=1):
        if self.count == self.window_size:
            pop_left_val = self.queue.popleft()
            self.sum -= pop_left_val
            self.count -= 1

        self.queue.append(val * weight)
        self.sum += val*weight
        self.count += 1
        self.avg = self.sum/self.count if self.count != 0 else 0

    def average(self):
        return self.avg

