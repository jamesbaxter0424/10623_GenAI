"""Health check of a running NYC training (run on AWS; prints one status line).

  python tools/nyc_watch.py ~/genai-final/logs/step4_baseline.log
Exit code: 0 = OK, 1 = ALERT (problem found), 2 = DONE (pipeline finished).
Checks: error / NaN in the log, tmux session gone before the end, log not updated
for 15 min, loss spike (last 3 windows vs median of the previous 30), grad-norm
spike, free disk < 5 GB.
"""
import os
import re
import shutil
import subprocess
import sys
import time

LINE = re.compile(r"\[sg_to_bev/(\w+)\] epoch (\d+)/\d+ iter (\d+)/(\d+) \| (.*) \| ([\d.]+) h")


def main(path):
    txt = open(path, errors="ignore").read().replace("\r", "\n")
    alerts = []
    if "BASELINE_DONE" in txt:
        print("DONE: pipeline finished (training, localization, generation, metrics)")
        return 2
    if re.search(r"Traceback|Error|out of memory", txt):
        alerts.append("error in log: " + re.findall(r".*(?:Traceback|Error|out of memory).*", txt)[-1][:160])
    if re.search(r"\bnan\b|\binf\b", txt, re.I):
        alerts.append("NaN/inf in losses")
    alive = subprocess.run(["tmux", "has-session", "-t", "train"], capture_output=True).returncode == 0
    if not alive:
        alerts.append("tmux session 'train' is gone but the pipeline did not finish")
    age = (time.time() - os.path.getmtime(path)) / 60
    if alive and age > 15:
        alerts.append(f"log not updated for {age:.0f} min (stalled?)")
    free = shutil.disk_usage("/").free / 1e9
    if free < 5:
        alerts.append(f"low disk: {free:.1f} GB free")

    rows = []
    for m in LINE.finditer(txt):
        vals = dict(kv.split(": ") for kv in m.group(5).split(", "))
        rows.append((m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                     {k: float(v) for k, v in vals.items()}, float(m.group(6))))
    status = "no training lines yet"
    if rows:
        mode, ep, it, n_it, v, hrs = rows[-1]
        stage = [r for r in rows if r[0] == mode]
        status = (f"{mode} epoch {ep} iter {it}/{n_it} | " +
                  ", ".join(f"{k} {x:.3f}" for k, x in v.items()) + f" | {hrs:.2f} h")
        if mode == "train" and len(stage) > 40:
            hist = sorted(r[4]["diffusion"] for r in stage[-33:-3])
            med = hist[len(hist) // 2]
            last = [r[4]["diffusion"] for r in stage[-3:]]
            if min(last) > 1.5 * med:
                alerts.append(f"loss spike: diffusion last 3 windows {last} vs median {med:.2f}")
            if "grad_norm" in v and ep > 1:
                g_hist = sorted(r[4]["grad_norm"] for r in stage[-33:-3])
                g_med = g_hist[len(g_hist) // 2]
                if v["grad_norm"] > max(5 * g_med, 10):
                    alerts.append(f"grad-norm spike: {v['grad_norm']:.1f} vs median {g_med:.2f}")
    ckpts = sorted(re.findall(r"Saved checkpoint: (\S+)", txt))
    status += f" | ckpt: {os.path.basename(ckpts[-1]) if ckpts else 'none'} | disk free {free:.0f} GB"
    if alerts:
        print("ALERT: " + " || ".join(alerts) + " | " + status)
        return 1
    print("OK: " + status)
    return 0


if __name__ == "__main__":
    sys.exit(main(os.path.expanduser(sys.argv[1])))
