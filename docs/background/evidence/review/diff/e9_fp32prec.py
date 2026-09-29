"""E9: the plan's ieee_matmul() context (torch.backends.cuda.matmul.fp32_precision) vs user code that uses
the legacy TF32 APIs (allow_tf32 / set_float32_matmul_precision), and thread-locality of the flag."""
import sys, threading, warnings, torch
warnings.simplefilter("always")
scenario = sys.argv[1]
def ieee_matmul_enter():
    old = torch.backends.cuda.matmul.fp32_precision
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    return old
def show(tag):
    for name, get in (("fp32_precision", lambda: torch.backends.cuda.matmul.fp32_precision),
                      ("allow_tf32", lambda: torch.backends.cuda.matmul.allow_tf32),
                      ("get_float32_matmul_precision", lambda: torch.get_float32_matmul_precision())):
        try:
            print(f"   [{tag}] {name} = {get()}")
        except Exception as e:
            print(f"   [{tag}] {name}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
if scenario == "legacy_user":
    print("user: torch.backends.cuda.matmul.allow_tf32 = True ; then library enters/leaves ieee_matmul()")
    torch.backends.cuda.matmul.allow_tf32 = True
    old = ieee_matmul_enter(); show("inside"); torch.backends.cuda.matmul.fp32_precision = old; show("after")
elif scenario == "set_prec_user":
    print("user: torch.set_float32_matmul_precision('high') ; then library enters/leaves ieee_matmul()")
    torch.set_float32_matmul_precision("high")
    old = ieee_matmul_enter(); show("inside"); torch.backends.cuda.matmul.fp32_precision = old; show("after")
elif scenario == "thread":
    print("is the flag thread-local? set 'ieee' in a worker thread, read in main thread")
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    t = threading.Thread(target=ieee_matmul_enter); t.start(); t.join()
    show("main after worker set ieee")
