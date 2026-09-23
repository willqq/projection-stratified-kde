"""Build the CPU posting extension using the active reproducibility environment."""
from pathlib import Path
import sysconfig,subprocess,pybind11,json,time
p=Path(__file__).resolve().parent
cmd=['c++','-O3','-shared','-std=c++14','-fPIC','-I'+pybind11.get_include(),'-I'+sysconfig.get_paths()['include'],str(p/'posting_index.cpp'),'-o',str(p/('_posting_index'+sysconfig.get_config_var('EXT_SUFFIX')))]
r=subprocess.run(cmd,capture_output=True,text=True)
(p/'posting_build.json').write_text(json.dumps({'command':cmd,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'timestamp':time.time()},indent=2))
if r.returncode:raise RuntimeError(r.stderr)
