class Client:
    def __init__(self, *args, **kwargs): pass
    def send(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass
    def join(self, *args, **kwargs): pass
    
def Client(*args, **kwargs):
    class C:
        def send(self, *args, **kwargs): pass
        def flush(self, *args, **kwargs): pass
        def join(self, *args, **kwargs): pass
    return C()
    
def flush(*args, **kwargs): pass
def join(*args, **kwargs): pass
def identify(*args, **kwargs): pass
def track(*args, **kwargs): pass
def page(*args, **kwargs): pass
def screen(*args, **kwargs): pass
def group(*args, **kwargs): pass
def alias(*args, **kwargs): pass
