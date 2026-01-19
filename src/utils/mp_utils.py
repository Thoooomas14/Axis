import multiprocessing

# Define NoDaemonProcess to allow nested multiprocessing
# (i.e. allow DataLoader workers to spawn their own subprocesses)
class NoDaemonProcess(multiprocessing.Process):
    """
    A process that can spawn children. 
    Required because PyTorch DataLoader workers are daemons by default,
    and daemons cannot spawn children (which SubprocessTFDSLoader needs).
    """
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass

class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess
