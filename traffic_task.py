#!/usr/bin/env python3
"""Non-AI traffic for the retry: real browser video and page browsing. Runs until killed."""
import sys, time
from playwright.sync_api import sync_playwright

kind, arg = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                          args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
    page = b.new_context().new_page()
    if kind == "video":
        page.goto(f"https://www.youtube.com/watch?v={arg}", timeout=20000)
        while True:
            page.evaluate("(()=>{const v=document.querySelector('video');if(v&&v.paused)v.play().catch(()=>{})})()")
            time.sleep(1)
    else:
        for url in arg.split("|"):
            try:
                page.goto(url, timeout=8000, wait_until="load")
            except Exception:
                pass
            time.sleep(3)
        time.sleep(600)
