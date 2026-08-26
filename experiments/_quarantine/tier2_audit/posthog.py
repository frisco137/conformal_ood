class Posthog:
    def __init__(self, *args, **kwargs): pass
    def capture(self, *args, **kwargs): pass
    def identify(self, *args, **kwargs): pass
    def group(self, *args, **kwargs): pass
    def flush(self, *args, **kwargs): pass
    def join(self, *args, **kwargs): pass
    def page(self, *args, **kwargs): pass
    
def Posthog(*args, **kwargs):
    class C:
        def capture(self, *args, **kwargs): pass
        def identify(self, *args, **kwargs): pass
        def flush(self, *args, **kwargs): pass
        def join(self, *args, **kwargs): pass
    return C()
    
def capture(*args, **kwargs): pass
def identify(*args, **kwargs): pass
def group(*args, **kwargs): pass
def flush(*args, **kwargs): pass
def join(*args, **kwargs): pass
def page(*args, **kwargs): pass
