#!/usr/bin/env python3
import argparse
import asyncio
import csv
import os
import re
import sys
import time
import glob
from collections import deque, defaultdict
from urllib.parse import urljoin, urlparse

import httpx
from bs4 import BeautifulSoup
from urllib import robotparser
from datetime import datetime

USER_AGENT = "EmailScraperPro/1.0 (+https://github.com/)"
CONNECT_TIMEOUT = 10.0
READ_TIMEOUT = 15.0

PRIORITY_PATHS = [
    "/", "/contact", "/kontakt", "/kontaktformular", "/contact-us", "/contacts",
    "/impressum", "/imprint", "/legal", "/privacy", "/datenschutz",
    "/about", "/ueber-uns", "/uber-uns", "/info", "/colophon", "/colofon"
]

EMAIL_REGEX = re.compile(
    r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I
)

SUSPECT_SUBSTRINGS = [
    "@lieferando", "@just-eat", "@justeat", "@takeaway", "@ubereats", "@wolt",
    "@opentable", "@deliveroo", "@doordash", "@citypantry", "@menulog",
    "@bookatable", "@resmio"
]

IGNORE_SUBSTRINGS = [
    "@example.", "no-reply@", "noreply@", "donotreply@", "@localhost", "@test.", "@invalid."
]

def now_ts():
    return datetime.utcnow().strftime("%H:%M:%S")

def norm_host(website: str) -> str:
    if not website:
        return ""
    w = website.strip()
    if not re.match(r"^https?://", w, re.I):
        w = "https://" + w.strip("/")
    try:
        p = urlparse(w)
        host = p.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""

def same_host(url: str, host: str) -> bool:
    try:
        h = urlparse(url).netloc.lower()
        if h.startswith("www."):
            h = h[4:]
        return h == host
    except Exception:
        return False

def build_url(base_host: str, path: str) -> str:
    if not path.startswith("/"):
        path = "/" + path
    return "https://" + base_host + path

def pick_website_column(headers):
    cand = ["website","url","domain","site","homepage"]
    for c in cand:
        for h in headers:
            if h.lower()==c:
                return h
    # fallback: first that includes 'web' or 'url'
    for h in headers:
        if "web" in h.lower() or "url" in h.lower():
            return h
    return headers[0]

def extract_emails_from_mailtos(soup) -> set:
    found = set()
    for a in soup.find_all("a", href=True):
        href = a.get("href","")
        if href.lower().startswith("mailto:"):
            addr = href.split(":",1)[1].split("?")[0].strip()
            if addr:
                found.add(addr)
    return found

def extract_emails_from_text(text: str) -> set:
    found = set()
    for m in EMAIL_REGEX.finditer(text):
        found.add(m.group(0))
    return found

def categorize_email(addr: str) -> tuple[bool, bool]:
    a = addr.lower()
    if any(x in a for x in IGNORE_SUBSTRINGS):
        return False, False  # drop
    suspect = any(x in a for x in SUSPECT_SUBSTRINGS)
    return True, suspect

def parse_html_links_and_emails(url: str, html: str):
    # Robust soup parse with fallback
    soup = None
    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        soup = BeautifulSoup(html, "html.parser")

    emails = set()
    emails |= extract_emails_from_mailtos(soup)
    emails |= extract_emails_from_text(soup.get_text(" ", strip=True))

    links = set()
    for a in soup.find_all("a", href=True):
        links.add(urljoin(url, a["href"]))

    # Also parse <footer> text aggressively
    for foot in soup.find_all("footer"):
        emails |= extract_emails_from_text(foot.get_text(" ", strip=True))

    return links, emails

async def fetch(client: httpx.AsyncClient, url: str, max_tries=2):
    last_exc = None
    for _ in range(max_tries):
        try:
            r = await client.get(url, timeout=httpx.Timeout(READ_TIMEOUT))
            if r.status_code >= 200 and r.status_code < 400:
                return r
        except Exception as e:
            last_exc = e
        await asyncio.sleep(0.2)
    raise last_exc if last_exc else RuntimeError("fetch failed")

async def get_robot_parser(client: httpx.AsyncClient, base: str):
    rp = robotparser.RobotFileParser()
    try:
        r = await client.get(f"https://{base}/robots.txt", timeout=httpx.Timeout(READ_TIMEOUT))
        if r.status_code == 200 and r.text:
            rp.parse(r.text.splitlines())
        else:
            rp.parse([])
    except Exception:
        rp.parse([])
    return rp

async def crawl_host(host: str, obey_robots: bool, max_pages: int, per_host_conc: int, log_cb):
    limits = httpx.Limits(max_connections=per_host_conc, max_keepalive_connections=per_host_conc)
    headers = {"User-Agent": USER_AGENT, "Accept-Encoding":"gzip, deflate, br"}
    async with httpx.AsyncClient(http2=True, limits=limits, headers=headers, follow_redirects=True) as client:
        rp = None
        if obey_robots:
            rp = await get_robot_parser(client, host)

        q = deque()
        seen = set()
        for p in PRIORITY_PATHS:
            u = build_url(host, p)
            q.append(u); seen.add(u)

        found_emails = set()  # {(email, source_url)}
        sem = asyncio.Semaphore(per_host_conc)

        async def worker():
            nonlocal found_emails
            while q and len(seen) <= max_pages and len(found_emails) < 5000:
                url = q.popleft()
                if obey_robots and rp and not rp.can_fetch(USER_AGENT, url):
                    continue
                async with sem:
                    try:
                        r = await fetch(client, url)
                        text = r.text if isinstance(r.text, str) else r.content.decode("utf-8","ignore")
                        links, emails = parse_html_links_and_emails(str(r.url), text)

                        # log and collect emails
                        for e in sorted(emails):
                            ok, suspect = categorize_email(e)
                            if not ok:
                                continue
                            found_emails.add((e, str(r.url), suspect))
                            log_cb(host, e, str(r.url), suspect)

                        # enqueue new links (same host only)
                        for L in links:
                            if same_host(L, host) and L not in seen:
                                seen.add(L)
                                q.append(L)
                                if len(seen) >= max_pages:
                                    break
                    except Exception:
                        pass

        workers = [asyncio.create_task(worker()) for _ in range(per_host_conc)]
        await asyncio.gather(*workers)

    # Return unique emails as list of tuples
    uniq = {}
    for e, src, sus in found_emails:
        uniq[(host, e)] = (host, e, src, 1 if sus else 0)
    return list(uniq.values())

async def phase(hosts, obey_robots, max_pages, per_host_conc, hosts_parallel, tag, checkpoint):
    print(f"{now_ts()} {tag.upper()} starting — hosts: {len(hosts)} | robots={int(obey_robots)} | hosts_parallel={hosts_parallel} per_host={per_host_conc}")
    # limit concurrent hosts
    sem_hosts = asyncio.Semaphore(hosts_parallel)
    results = []
    found_counter = 0

    async def run_one(h):
        nonlocal found_counter
        async with sem_hosts:
            print(f"{now_ts()} START {tag.upper()} {h}")
            def logger(host, email, src, suspect):
                nonlocal found_counter
                found_counter += 1
                status = "[suspect]" if suspect else ""
                print(f"{now_ts()} [{tag}] {host} -> {email} ({src}) {status}")
                if checkpoint and (found_counter % checkpoint == 0):
                    checkpoint_flush()

            res = await crawl_host(h, obey_robots, max_pages, per_host_conc, logger)
            print(f"{now_ts()} DONE  {tag.upper()} {h} -> {len(res)} emails")
            results.extend(res)

    chunk_rows = []  # accumulate rows to write on checkpoint
    out_dir = CURRENT_RUN_DIR
    chunks_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    def checkpoint_flush():
        # spill chunk_rows to a chunk file
        nonlocal chunk_rows
        if not chunk_rows:
            return
        ts = datetime.utcnow().strftime("%H%M%S")
        path = os.path.join(chunks_dir, f"emails_chunk_{ts}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["host","email","source_url","suspect"])
            w.writerows(chunk_rows)
        print(f"{now_ts()} CHECKPOINT saved -> {path} ({len(chunk_rows)} rows)")
        chunk_rows = []

    async def collector():
        # intercept additions to 'results' to stage chunk writes
        last_len = 0
        while True:
            await asyncio.sleep(0.5)
            if STOP_COLLECTOR:
                break
            if len(results) > last_len:
                # stage the delta for checkpoint chunk
                for row in results[last_len:]:
                    chunk_rows.append(row)
                last_len = len(results)

    collector_task = asyncio.create_task(collector())
    STOP_COLLECTOR = False
    try:
        await asyncio.gather(*(run_one(h) for h in hosts))
    finally:
        STOP_COLLECTOR = True
        await collector_task
        checkpoint_flush()

    # dedupe results
    uniq = {}
    for r in results:
        host, email, src, suspect = r
        uniq[(host,email)] = r
    return list(uniq.values())

def read_hosts_from_csv(path, website_col=None):
    hosts = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        headers = r.fieldnames or []
        col = website_col or pick_website_column(headers)
        for row in r:
            host = norm_host(row.get(col, ""))
            if host:
                hosts.append(host)
    # dedupe while preserving order
    seen = set()
    ordered = []
    for h in hosts:
        if h not in seen:
            seen.add(h)
            ordered.append(h)
    return ordered

def write_final_outputs(run_dir, all_rows, hosts_all):
    # final emails
    final_csv = os.path.join(run_dir, "emails_found.csv")
    with open(final_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["host","email","source_url","suspect"])
        w.writerows(all_rows)

    # no_email list
    found_hosts = {h for (h, *_ ) in all_rows}
    no_hosts = [h for h in hosts_all if h not in found_hosts]
    no_csv = os.path.join(run_dir, "no_email.csv")
    with open(no_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["host"])
        for h in no_hosts:
            w.writerow([h])

    # archive chunks into subdir 'chunks' (already there)
    print(f"{now_ts()} Finalized -> {final_csv} ({len(all_rows)} rows)")
    print(f"{now_ts()} No-email hosts -> {no_csv} ({len(no_hosts)} hosts)")
    return final_csv, no_csv

def main():
    parser = argparse.ArgumentParser(description="Async email scraper (two-phase) with checkpointing.")
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--website_col", default=None)
    parser.add_argument("--obey_robots", type=int, default=1)
    parser.add_argument("--hosts_parallel", type=int, default=8)
    parser.add_argument("--per_host", type=int, default=4)
    parser.add_argument("--max_pages_shallow", type=int, default=40)
    parser.add_argument("--max_pages_deep", type=int, default=150)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    args = parser.parse_args()

    global CURRENT_RUN_DIR
    run_tag = datetime.utcnow().strftime("run-%Y-%m-%d_%H%M")
    out_dir = os.path.join("out", run_tag)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output dir: {out_dir}")
    CURRENT_RUN_DIR = out_dir

    # log file
    log_path = os.path.join(out_dir, "scrape.log")
    sys.stdout = open(log_path, "w", buffering=1, encoding="utf-8")
    sys.stderr = sys.stdout
    print(f"{now_ts()} START (robots={args.obey_robots})")

    # also tee to console
    class Tee:
        def __init__(self, *streams):
            self.streams = streams
        def write(self, data):
            for s in self.streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass
        def flush(self):
            for s in self.streams:
                try:
                    s.flush()
                except Exception:
                    pass

    console = sys.__stdout__
    sys.stdout = Tee(sys.stdout, console)
    sys.stderr = sys.stdout

    hosts = read_hosts_from_csv(args.input_csv, args.website_col)
    print(f"{now_ts()} total hosts: {len(hosts)}")

    # Phase 1: shallow
    shallow_rows = asyncio.run(phase(
        hosts, bool(args.obey_robots), args.max_pages_shallow, args.per_host,
        args.hosts_parallel, "shallow", args.checkpoint_every
    ))

    # Determine which hosts need deep phase
    have_email_hosts = {h for (h, *_ ) in shallow_rows}
    pending = [h for h in hosts if h not in have_email_hosts]
    print(f"{now_ts()} shallow found emails for {len(have_email_hosts)} hosts; pending deep: {len(pending)}")

    # Phase 2: deep (only for pending)
    deep_rows = []
    if pending:
        deep_rows = asyncio.run(phase(
            pending, bool(args.obey_robots), args.max_pages_deep, args.per_host,
            args.hosts_parallel, "deep", args.checkpoint_every
        ))

    # Merge + dedupe
    all_rows = {}
    for r in shallow_rows + deep_rows:
        host, email, src, suspect = r
        all_rows[(host, email)] = r
    final_rows = sorted(all_rows.values(), key=lambda x: (x[0], x[1]))

    # Final outputs
    write_final_outputs(out_dir, final_rows, hosts)

if __name__ == "__main__":
    main()
