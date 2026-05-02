class SuspensionHead:
    def forward(self, x):
        raise NotImplementedError


class SuspensionMemory:
    def forward(self, x, memory_state=None):
        raise NotImplementedError
