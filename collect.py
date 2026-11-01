#!/usr/bin/env python3
"""Run AI-agent tasks over this laptop's own Wi-Fi and record packet metadata only.

Keeps: time, length, direction. Headers are truncated at capture (64 bytes) and
the pcap is deleted after extraction, so no content or IP addresses are stored.

Every run records a fixed window, so run length cannot give the class away.
Order is shuffled, so time of day cannot give the class away.
"""
import argparse, csv, json, os, random, signal, subprocess, tempfile, time
from pathlib import Path

IFACE = "wlp0s20f3"
LOCAL_IP = LOCAL_MAC = None  # filled at start; direction comes from MAC so IPv6 counts
HERE = Path(__file__).parent
ROOT = Path(os.environ.get("PILOT_DIR", HERE))
ROOT.mkdir(exist_ok=True)
OUT = ROOT / "data"

TOPICS = ["tides", "bridges", "bees", "volcanoes", "glass", "compasses", "rainbows", "coral",
          "lighthouses", "yeast", "glaciers", "magnets", "paper", "owls", "salt", "clocks"]
SMALL_CODE = ["a word counter", "a prime sieve", "a matrix transpose", "a date parser", "an LRU cache",
              "a run-length encoder", "a binary search", "a CSV column summer", "a roman numeral converter",
              "a queue using two stacks", "a Levenshtein distance", "a base64 encoder", "a topological sort",
              "a bracket checker", "a histogram printer", "an interval merger"]

CLASSES = {
    "chat": dict(prompt="Write a detailed 400-word explanation of {x}. No tools.", arg=TOPICS, tools=""),
    "code": dict(prompt="In this folder write {x} in Python as a file with three tests, run the tests, fix any failure, then stop.",
                 arg=SMALL_CODE, tools="Write,Edit,Read,Bash(python3:*)"),
    "web": dict(prompt="Use web search and fetch two pages to find recent news about {x}. Give three short bullets.",
                arg=TOPICS, tools="WebSearch,WebFetch"),
    "idle": dict(prompt=None, arg=TOPICS, tools=""),  # control: no agent at all
    "video": dict(prompt=None, tools="", arg=["aqz-KE-bpKQ", "YE7VzlLtp-4", "eRsGyueVLvQ", "TLkA0RELQ1g"]),
    "browse": dict(prompt=None, tools="", arg=[
        "https://en.wikipedia.org/wiki/Tide|https://www.bbc.com/news|https://www.gutenberg.org/|https://www.nasa.gov/",
        "https://en.wikipedia.org/wiki/Bridge|https://www.reuters.com/|https://arxiv.org/|https://www.who.int/",
        "https://en.wikipedia.org/wiki/Bee|https://www.aljazeera.com/|https://www.python.org/|https://www.un.org/",
        "https://en.wikipedia.org/wiki/Glass|https://www.dawn.com/|https://github.com/|https://www.nature.com/"]),
    # capped so one run uses at most about 10 MB
    "download": dict(prompt=None, tools="", arg=["256K", "512K"]),
}


if os.environ.get("PROMPT_SET") == "2":  # confirmation session: wording and topics not used before
    T2 = ["sundials", "kelp", "granite", "windmills", "fireflies", "aqueducts", "peat", "barometers",
          "lichen", "canals", "sonar", "beeswax", "quartz", "typewriters", "mangroves", "ferries"]
    C2 = ["a stopword remover", "a fizzbuzz variant", "a temperature converter", "a ring buffer", "a trie",
          "an anagram grouper", "a quicksort", "a JSON key flattener", "a caesar cipher", "a min heap",
          "a longest common prefix", "a hex dump printer", "a dijkstra shortest path", "a palindrome finder",
          "a moving average", "a time duration parser"]
    CLASSES["chat"].update(arg=T2, prompt="Give me a thorough plain-language overview of {x} in roughly 400 words, without using any tools.")
    CLASSES["code"].update(arg=C2, prompt="Create {x} as a Python module here, add a small test script, execute it, repair anything broken, and finish.")
    CLASSES["web"].update(arg=T2, prompt="Look online for what has been reported lately on {x}, read a couple of sources, and summarise in three lines.")


def local_ip():
    out = subprocess.check_output(["ip", "-4", "-o", "addr", "show", IFACE], text=True)
    return out.split()[3].split("/")[0]


def run_once(run_id, cls, arg, window, model):
    with tempfile.TemporaryDirectory(prefix="agentrun_") as work:
        pcap = Path(work) / "c.pcapng"
        cap = subprocess.Popen(["dumpcap", "-q", "-i", IFACE, "-s", "64", "-w", str(pcap)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(2)
        proc, t0, rc, finished = None, time.time(), None, None
        c = CLASSES[cls]
        if c["prompt"]:
            env = {k: v for k, v in os.environ.items() if not k.startswith("CLAUDE")}
            cmd = ["claude", "-p", c["prompt"].format(x=arg), "--model", model,
                   "--allowedTools", c["tools"], "--no-session-persistence"]
            proc = subprocess.Popen(cmd, cwd=work, env=env, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                    start_new_session=True)
        elif cls in ("video", "browse"):
            proc = subprocess.Popen(["python3", str(HERE / "traffic_task.py"), cls, arg], cwd=work,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        elif cls == "download":
            proc = subprocess.Popen(["curl", "-s", "-o", "/dev/null", "--limit-rate", arg,
                                     "https://speed.cloudflare.com/__down?bytes=40000000"], cwd=work,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        while time.time() - t0 < window:
            if proc and rc is None:
                rc = proc.poll()
                if rc is not None:
                    finished = time.time() - t0
            time.sleep(0.2)
        if proc and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        time.sleep(1)
        cap.send_signal(signal.SIGINT)
        cap.wait(timeout=15)
        fields = subprocess.run(
            ["tshark", "-r", str(pcap), "-T", "fields", "-E", "separator=,",
             "-e", "frame.time_epoch", "-e", "frame.len", "-e", "eth.src", "-e", "eth.dst"],
            capture_output=True, text=True).stdout
        rows = []
        for line in fields.splitlines():
            p = line.split(",")
            if len(p) < 4 or not p[0]:
                continue
            t = float(p[0]) - t0
            if 0 <= t <= window:
                rows.append((round(t, 6), int(p[1]), "down" if p[3] == LOCAL_MAC else "up" if p[2] == LOCAL_MAC else "other"))
        path = OUT / f"{run_id:04d}_{cls}.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t", "len", "dir"]); w.writerows(rows)
        # the pcap goes with the temp folder
    return dict(run=run_id, cls=cls, arg=arg, start=t0, window=window, agent_rc=rc,
                agent_seconds=finished, packets=len(rows), file=path.name)


def main():
    global LOCAL_IP, LOCAL_MAC
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=12)
    ap.add_argument("--window", type=int, default=60)
    ap.add_argument("--classes", default="chat,code,web,idle")
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--start-id", type=int, default=0)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    LOCAL_IP = local_ip()
    LOCAL_MAC = Path(f"/sys/class/net/{IFACE}/address").read_text().strip()
    rng = random.Random(a.seed)
    plan = []
    for cls in a.classes.split(","):
        args = list(CLASSES[cls]["arg"]); rng.shuffle(args)
        plan += [(cls, args[i % len(args)]) for i in range(a.per_class)]
    rng.shuffle(plan)
    with open(ROOT / "runs.jsonl", "a") as log:
        for i, (cls, arg) in enumerate(plan, a.start_id):
            r = run_once(i, cls, arg, a.window, a.model)
            log.write(json.dumps(r) + "\n"); log.flush()
            print(r, flush=True)
            time.sleep(3)


if __name__ == "__main__":
    main()
