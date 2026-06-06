import sys, time
print(sys.executable)
sys.stdout.flush()
print('PY_START', time.time())
sys.stdout.flush()
try:
    import streamlit
    print('STREAMLIT', getattr(streamlit, '__version__', 'unknown'), time.time())
except Exception as e:
    print('IMPORT_ERROR', type(e).__name__, str(e), time.time())

sys.stdout.flush()
