# Windows — offline deployment, step by step

A beginner-friendly walkthrough to run Sainik Sahayak on a **Windows** machine
that will hold your restricted PDF, kept off the network.

The approach that is easiest **and** sound: **install everything while the
machine has internet, then permanently disconnect it from all networks _before_
the restricted PDF ever goes on it.** The PDF only ever exists on the machine
after it is offline for good. (If your security rules require a machine that has
*never* touched the internet, use the stricter staging path in `README.md §4–5`
instead — the same scripts, with a USB transfer.)

Take it one part at a time. If any step errors, copy the error back to me and
I'll unstick you.

---

## What you need

- A Windows 10/11 machine (64-bit) you can dedicate to this.
- **16 GB RAM recommended** (8 GB works with a smaller model — see Part 3),
  and **~40 GB free disk**.
- Administrator rights on it.
- Your restricted PDF (kept aside for now — Part 7).

---

## Part 1 — Install the tools (while online)

1. Open **PowerShell as Administrator** (Start → type "PowerShell" → right-click
   → Run as administrator) and run:
   ```powershell
   wsl --install
   ```
   This installs WSL2 + Ubuntu (a Linux environment inside Windows). **Reboot**
   when it asks.

2. After reboot, **Ubuntu** opens and asks you to create a username and
   password. Pick any — remember the password (you'll use it for `sudo`).

3. Install **Docker Desktop for Windows**: download from
   `https://www.docker.com/products/docker-desktop/`, run the installer, and
   keep **"Use WSL 2 based engine"** checked. After it installs, open Docker
   Desktop → **Settings → Resources → WSL Integration** → turn **on** the toggle
   for your Ubuntu distro. Click **Apply & Restart**.

4. Open the **Ubuntu** app and check both tools work:
   ```bash
   docker version
   docker compose version
   ```
   Both should print versions (no "cannot connect" error). If Docker errors,
   make sure Docker Desktop is running and WSL integration is on (step 3).

---

## Part 2 — Get the project files (while online)

Do everything below **inside the Ubuntu window**, in your home folder.

1. Install git (openssl and curl are already there):
   ```bash
   sudo apt update && sudo apt install -y git
   ```

2. Get this project and switch to the right branch:
   ```bash
   cd ~
   git clone https://github.com/mannpandya1702/chatbot.git
   cd chatbot
   git checkout claude/repository-review-5ox08x
   ```
   (If git asks you to log in, use your GitHub username and a **personal access
   token** as the password — GitHub no longer accepts your account password on
   the command line.)

3. Get the Supabase self-hosting bundle:
   ```bash
   cd ~
   git clone --depth 1 https://github.com/supabase/supabase.git
   ```
   Its folder for us is `~/supabase/docker`.

4. Make a folder for your PDF (leave it empty for now):
   ```bash
   mkdir -p ~/kb
   ```

---

## Part 3 — Generate the configuration (while online)

This makes all passwords and security keys for you — you don't edit secrets by
hand.

```bash
cd ~/chatbot
SUPABASE_DOCKER_DIR=~/supabase/docker \
SUPABASE_PUBLIC_URL=http://localhost:8000 \
GEN_MODEL=qwen2.5:7b-instruct \
KB_DIR=$HOME/kb \
  bash deploy/airgap/bootstrap-env.sh
```

- Using the machine at its own screen? `http://localhost:8000` is correct.
- **Only 8 GB RAM?** Change `GEN_MODEL=qwen2.5:7b-instruct` to
  `GEN_MODEL=qwen2.5:3b-instruct` (smaller, lighter model).

---

## Part 4 — Bring it all up (while online)

One command. It pulls the Supabase images, builds our two images, starts
everything, sets up the database, creates your admin login, and downloads the
language model. **First run takes a while** (several GB of downloads).

```bash
cd ~/chatbot
SUPABASE_DOCKER_DIR=~/supabase/docker \
  bash deploy/airgap/bringup.sh --slim --no-load
```

Near the end it prints your **admin login and a one-time password** —
**copy those somewhere safe now.** It ends with "Sainik Sahayak is up."

Open a browser on the machine to **http://localhost:3000** — you should reach
the app. (The chat/login screens are still being built — Part 8 notes this.)

---

## Part 5 — (Recommended) test with a throwaway file, then clean up

Prove it works *before* you put the real PDF on. Make any harmless test PDF
(e.g. print a web page to PDF), drop it in `~/kb`, and:

```bash
cd ~/supabase/docker
docker compose --env-file .env \
  -f docker-compose.yml \
  -f ~/chatbot/deploy/airgap/docker-compose.airgap.yml \
  exec rag python cli.py ingest /kb --tier 1
```

You should see `OK <file>.pdf: N pages, N chunks`. When happy, remove the test
doc and its data (so only your real content will exist later):

```bash
docker exec -i supabase-db psql -U postgres -d postgres -c "delete from public.documents;"
rm ~/kb/*.pdf
```

---

## Part 6 — Disconnect from the network (before the PDF goes on)

Now take the machine offline **for good**:

- Turn off Wi-Fi and unplug the Ethernet cable, **or**
- Disable the network adapters (Settings → Network & internet → advanced).

Everything from here runs locally — the language model, the database, the
search. Nothing leaves the machine.

---

## Part 7 — Add your PDF and ingest it

1. Copy your restricted PDF into the `~/kb` folder. From the Ubuntu window you
   can reach your Windows files under `/mnt/c/...`, e.g.:
   ```bash
   cp "/mnt/c/Users/<YourWindowsName>/Documents/pamphlet.pdf" ~/kb/
   ```

2. Ingest it. Choose the access tier: **1** = every jawan can see it, **2**/**3**
   = progressively more restricted.
   ```bash
   cd ~/supabase/docker
   docker compose --env-file .env \
     -f docker-compose.yml \
     -f ~/chatbot/deploy/airgap/docker-compose.airgap.yml \
     exec rag python cli.py ingest /kb --tier 2
   ```
   Scanned pages are OCR'd (Hindi + English) automatically; this can take a few
   minutes for a big document.

---

## Part 8 — Use it

- Open **http://localhost:3000** and sign in with the admin login from Part 4.
  First login forces a password change and TOTP setup (an authenticator app
  like Google Authenticator).
- **Note:** the browser chat and login screens are the next thing to build
  (Phases 3–4). Until they land, you can confirm answers from the command line —
  ask me and I'll give you the exact `curl` test for your document. Once I build
  the UI, asking questions happens in the browser.

---

## Everyday operations

- **Stop everything:** `cd ~/supabase/docker && docker compose --env-file .env -f docker-compose.yml -f ~/chatbot/deploy/airgap/docker-compose.airgap.yml down`
- **Start again later:** re-run the Part 4 command (it's safe to re-run — it
  won't wipe data or re-create your admin).
- **Add more documents:** drop PDFs in `~/kb`, re-run the Part 7 ingest.

## Gotchas

- **Docker Desktop after going offline:** installed containers run fine offline.
  Docker Desktop may show sign-in prompts — you can ignore them; you don't need
  to log in to run the stack.
- **"no space left" / slow model:** the 7B model needs real RAM. If generation
  is very slow or the machine struggles, redo Part 3 with `qwen2.5:3b-instruct`
  and re-run Part 4.
- **Keep the machine's clock roughly correct** — TOTP (the 6-digit login codes)
  depends on the time being right.
