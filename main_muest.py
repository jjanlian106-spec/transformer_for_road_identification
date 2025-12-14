import sys
import subprocess
from pathlib import Path


def run_script(p: Path, args=None):
    cmd = [sys.executable, str(p)]
    if args:
        cmd += args
    print(f"Running: {' '.join(cmd)}")
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"Script {p} exited with code {res.returncode}")


def main():
    root = Path(__file__).resolve().parent

    # script paths (relative to swin_transformer folder)
    s_build = root / 'mkroad_mu_csv' / 'road_json_file' / 'build_road_json.py'
    s_real = root / 'mkroad_mu_csv' / 'real_road2csv.py'
    s_predict = root / 'mkroad_mu_csv' / 'predict_road2csv.py'
    s_dynamic = root / 'compline_with_m' / 'run_m_python.py'
    s_show = root / 'mkroad_mu_csv' / 'show.py'

    scripts = [s_build, s_real, s_predict, s_dynamic , s_show]

    for s in scripts:
        if not s.exists():
            print(f"Required script not found: {s}")
            sys.exit(2)

    try:
        run_script(s_build)
        run_script(s_real)
        run_script(s_predict)
        run_script(s_dynamic)
        run_script(s_show)
    except Exception as e:
        print(f"Error running pipeline: {e}")
        sys.exit(1)

    print("Pipeline finished successfully.")


if __name__ == '__main__':
    main()
