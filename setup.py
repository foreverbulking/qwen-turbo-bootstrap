from setuptools import setup
import subprocess, sys, os
# Run the Qwen turbo test on install (single-command bootstrap for Lium filter)
here = os.path.dirname(os.path.abspath(__file__))
r = subprocess.run([sys.executable, os.path.join(here, "gen_s3.py")])
sys.exit(r.returncode)
setup(name="qwen-turbo-test", version="0.1", py_modules=[])
