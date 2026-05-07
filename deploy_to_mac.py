"""
将 causal_demo 项目部署到远程 Mac 并运行测试.
通过 mac-r 的 HTTP API 发送命令.
"""

import requests
import base64
import time
import sys
import os
import json

SERVER = "http://localhost:8900"
REMOTE_DIR = "/Users/bytedance/Desktop/hehe/research/causal_demo"

def send_cmd(cmd, timeout=60):
    """发送命令到远程 Mac 并等待结果"""
    requests.post(f"{SERVER}/cmd", json={"cmd": cmd})
    time.sleep(2)

    for _ in range(timeout):
        resp = requests.get(f"{SERVER}/results")
        results = resp.json()
        if results:
            last = results[-1]
            if last.get("cmd", "").strip() == cmd.strip():
                return last
        time.sleep(1)
    return None

def send_file(local_path, remote_path):
    """通过 base64 编码传输文件到远程 Mac"""
    with open(local_path, "r", encoding="utf-8") as f:
        content = f.read()
    encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
    cmd = f"echo '{encoded}' | base64 --decode > {remote_path}"
    return send_cmd(cmd)

def main():
    print("=" * 60)
    print("部署 causal_demo 到远程 Mac")
    print("=" * 60)

    # 1. 创建远程目录
    print("\n[1/4] 创建远程目录...")
    r = send_cmd(f"mkdir -p {REMOTE_DIR}/output")
    if r:
        print(f"  ✓ 目录已创建 (exit code: {r.get('code', '?')})")

    # 2. 传输文件
    files = [
        "requirements.txt",
        "01_ate_simulation.py",
        "02_propensity_score.py",
        "03_instrumental_variables.py",
        "04_difference_in_differences.py",
        "05_regression_discontinuity.py",
        "06_dag_and_do_calculus.py",
        "07_dowhy_demo.py",
    ]

    print(f"\n[2/4] 传输 {len(files)} 个文件...")
    local_dir = os.path.dirname(os.path.abspath(__file__))
    for f in files:
        local_path = os.path.join(local_dir, f)
        remote_path = f"{REMOTE_DIR}/{f}"
        r = send_file(local_path, remote_path)
        status = "✓" if r and r.get("code") == 0 else "✗"
        print(f"  {status} {f}")

    # 3. 安装依赖
    print(f"\n[3/4] 安装 Python 依赖...")
    r = send_cmd(f"cd {REMOTE_DIR} && pip3 install -r requirements.txt", timeout=120)
    if r:
        code = r.get("code", -1)
        if code == 0:
            print(f"  ✓ 依赖安装完成")
        else:
            print(f"  ✗ 安装失败: {r.get('stderr', '')[:200]}")

    # 4. 运行 demo
    print(f"\n[4/4] 运行 demo 测试...")
    demo = sys.argv[1] if len(sys.argv) > 1 else "01_ate_simulation.py"
    print(f"  运行: {demo}")
    r = send_cmd(f"cd {REMOTE_DIR} && python3 {demo}", timeout=120)
    if r:
        print(f"\n--- 输出 ---")
        print(r.get("stdout", "")[:3000])
        if r.get("stderr"):
            print(f"\n--- 错误 ---")
            print(r.get("stderr", "")[:1000])
        print(f"\n  Exit code: {r.get('code', '?')}")


if __name__ == "__main__":
    main()
