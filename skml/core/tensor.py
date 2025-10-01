import numpy as np

class tensor:
    def __init__(self, data, dtype=None, device='cpu'):
        # Convert input to numpy array
        if isinstance(data, tensor):
            data = data.data
        self.data = np.array(data, dtype=dtype)


        self.dtype = self.data.dtype
        self.device = device

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"tensor(data={self.data}, dtype={self.dtype}, device='{self.device}')"
