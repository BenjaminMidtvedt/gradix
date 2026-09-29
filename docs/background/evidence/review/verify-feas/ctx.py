import contextvars, contextlib
H = contextvars.ContextVar("handler", default="none")
@contextlib.contextmanager
def intervene(v):
    tok = H.set(v)
    try: yield
    finally: H.reset(tok)
def stream():
    while True: yield H.get()          # a batch rendered under whatever handler is visible at next()
with intervene("noise_off"):
    it = stream(); first = next(it)     # created and first batch inside the context
second = next(it)                       # iterated after leaving the context
with intervene("noise_off"):
    it2 = stream()
print("inside:", first, "| after leaving:", second, "| created inside, iterated outside:", next(it2))
