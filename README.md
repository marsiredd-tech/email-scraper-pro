# Email Scraper Pro 🚀

A fast, async email crawler that finds likely **contact emails** from restaurant (or any business) websites.  
It prioritizes contact/legal pages, explores the whole site if needed, **saves checkpoints every N emails** to avoid data loss, and merges everything into a final CSV when the run completes.  
It also **flags possible bait/third‑party emails** (e.g., `@lieferando`, `@just-eat`) as *suspect* instead of mixing them with primary contacts.

---

## Features ✨
- **Two-phase crawl**: shallow (contact/legal/footers) ➜ deep (full-site) only for hosts where nothing was found
- **Async & parallel** crawling with per-host concurrency caps
- **Robots-aware** mode (can be disabled)
- **Robust parsing** with graceful fallback when HTML is malformed
- **Email extraction** from text and `mailto:` links
- **Bait filter**: flags emails from common aggregators/delivery platforms as `suspect=1`
- **Checkpointing** every N emails + final merge into `emails_found.csv`
- **Crash-safe**: partial CSV chunks are retained if the run stops early

---

## Quickstart 🧑‍🍳

### 1) Requirements
- Python **3.10+** (tested on 3.10–3.13)
- Recommended inside a virtualenv

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### 2) Prepare your input CSV
Place a CSV under `data/` with a column like `website`, `url`, `domain`, or similar.  
Example: `data/sample_restaurants.csv`

```csv
name,website
Sample Bistro,example.com
Another Place,https://example.org
```

### 3) Run 🏃‍♂️
```bash
python scripts/allinone_fast.py   --input_csv data/sample_restaurants.csv   --obey_robots 1   --hosts_parallel 12   --per_host 6   --max_pages_shallow 50   --max_pages_deep 200   --checkpoint_every 100
```

### Output 📦
All results are written into a fresh run directory like `out/run-YYYY-MM-DD_HHMM/`:
- `emails_found.csv` — final deduped results with columns: `host,email,source_url,suspect`
- `no_email.csv` — hosts where nothing could be found
- `chunks/*.csv` — checkpoint files (kept if the run crashed; archived into `chunks/` if completed)
- `scrape.log` — progress log

---

## CLI Options ⚙️
```
--input_csv PATH          Input CSV file (required)
--website_col NAME        Column name for website/url (auto-detects if omitted)
--obey_robots {0,1}       Respect robots.txt (default: 1)
--hosts_parallel N        Number of hosts to crawl in parallel (default: 8)
--per_host N              Max concurrent requests per host (default: 4)
--max_pages_shallow N     Max pages per host in shallow phase (default: 40)
--max_pages_deep N        Max pages per host in deep phase (default: 150)
--checkpoint_every N      Save chunk CSV every N found emails (default: 100)
```

---

## Notes & Tips 💡
- If performance is too low, increase `--hosts_parallel` and `--per_host` cautiously.
- If you get HTTP/2 errors, ensure `httpx[http2]` is installed (it is in `requirements.txt`).
- Some websites block bots aggressively; toggle `--obey_robots 0` at your own risk and responsibility.
- This tool **only** crawls within each site’s own domain (no external link following).

---

## License 📄
MIT — see `LICENSE`.
