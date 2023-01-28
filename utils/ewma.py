class EWMA:
    """
    Exponentially weighted moving average
    """
    def __init__(self, momentum=0.98):
        # set params
        self._running_val = None
        self._momentum = momentum

    def update(self, new_val):
        # update running val
        if self._running_val is not None:
            self._running_val = self._momentum * self._running_val + (1.0 - self._momentum) * new_val
        else:
            self._running_val = new_val

    @property
    def running_val(self):
        return self._running_val
