"""Start one coastal live run in the user's own AFSIM and connect the network."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from gateway.naming import serve
from gateway.prepare import prepare


def main() -> None:
    parser = argparse.ArgumentParser(description="用自己的 AFSIM 跑某临海城市这场，并接上神经网络。")
    parser.add_argument("--afsim", default=os.environ.get("AFSIM_ROOT", ""), help="AFSIM 2.9 安装目录，里面要有 bin/warlock.exe")
    parser.add_argument("--red", default="plans/fixed.json", help="红方方案。默认是固定阵地，不画航线。")
    parser.add_argument("--blue", default="plans/leap.json", help="蓝方方案。默认跃起俯冲撞击。")
    parser.add_argument("--work", default="", help="这场输出目录。默认写到 runs/live-时间。")
    parser.add_argument("--policy", default="builtin", help="builtin 用种子 3。换成自己的模型时写 模块:函数。")
    args = parser.parse_args()
    root = Path(args.afsim)
    warlock = root / "bin" / "warlock.exe"
    mission = root / "bin" / "mission.exe"
    if warlock.is_file():
        command = [str(warlock), "-console", "run.txt"]
    elif mission.is_file():
        command = [str(mission), "run.txt"]
    else:
        raise SystemExit("没有找到 bin/warlock.exe。把 AFSIM 2.9 的安装目录传给 --afsim，或设置环境变量 AFSIM_ROOT。")
    work = prepare(
        (ROOT / args.red).resolve() if not Path(args.red).is_absolute() else Path(args.red),
        (ROOT / args.blue).resolve() if not Path(args.blue).is_absolute() else Path(args.blue),
        Path(args.work) if args.work else None,
    )
    print(f"这场目录：{work}")
    print("Warlock 窗口就是这场仿真。关掉窗口，或在这个终端按 Ctrl+C，都会停。")
    host = subprocess.Popen(command, cwd=work)
    stop = threading.Event()

    def watch() -> None:
        host.wait()
        stop.set()

    threading.Thread(target=watch, daemon=True).start()
    try:
        serve(work, args.policy, stop)
    except KeyboardInterrupt:
        stop.set()
    finally:
        if host.poll() is None:
            host.terminate()


if __name__ == "__main__":
    main()
