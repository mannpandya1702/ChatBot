# Put it online — jawans reach it from anywhere

This gets Sainik Sahayak onto a server with a real domain and HTTPS, so anyone
with a login can use it from any network — on duty, on leave, on their phone.
No VPN, no unit LAN, nothing for jawans to install.

Everything runs on **one server that you control**: the database, the search, and
the language model. Nothing is sent to any outside service.

Offline/on-premise install instead? See [`../airgap/WINDOWS.md`](../airgap/WINDOWS.md).

---

## What you need

- **A server**: Ubuntu 22.04/24.04, **16 GB RAM**, 4 vCPU, ~80 GB disk, a public
  IP. Any provider works — pick one in the jurisdiction your client requires.
- **A domain** you can add DNS records to.
- Ports **80** and **443** open.

> 16 GB is the comfortable size: the search models need ~5 GB and the language
> model ~6 GB. 8 GB works if you use a smaller model (see Step 3).

---

## 1. Point two DNS records at the server

Both are needed — the second one is what the sign-in screen talks to. Replace the
IP with your server's.

| Type | Name | Value |
|------|------|-------|
| A | `sahayak` | `203.0.113.10` |
| A | `api.sahayak` | `203.0.113.10` |

That gives you `sahayak.example.in` and `api.sahayak.example.in`. **Wait until
both resolve before Step 5** — certificates are issued by checking DNS, so an
unresolved name fails there.

```bash
dig +short sahayak.example.in        # should print your server IP
dig +short api.sahayak.example.in
```

---

## 2. Install Docker and get the code

SSH into the server, then:

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && exec su -l $USER   # re-login so docker works without sudo

cd ~
git clone https://github.com/mannpandya1702/chatbot.git
git clone --depth 1 https://github.com/supabase/supabase.git
mkdir -p ~/kb
```

---

## 3. Generate the configuration

This writes every password and key for you. Use **https** and the **api**
hostname here:

```bash
cd ~/chatbot
SUPABASE_DOCKER_DIR=~/supabase/docker \
SUPABASE_PUBLIC_URL=https://api.sahayak.example.in \
GEN_MODEL=qwen2.5:7b-instruct \
KB_DIR=$HOME/kb \
  bash deploy/airgap/bootstrap-env.sh
```

**Only 8 GB RAM?** Use `GEN_MODEL=qwen2.5:3b-instruct` instead — faster, slightly
weaker answers.

---

## 4. Add the four hosted settings

```bash
cd ~/supabase/docker
cat >> .env <<'EOF'

# --- public HTTPS ---
APP_DOMAIN=sahayak.example.in
API_DOMAIN=api.sahayak.example.in
SITE_URL=https://sahayak.example.in
CADDYFILE=/home/YOUR_USER/chatbot/deploy/hosted/Caddyfile
WEB_PORT=127.0.0.1:3000
EOF
```

Edit the file to put in your real domains and your actual home path
(`echo $HOME` if unsure).

`WEB_PORT=127.0.0.1:3000` keeps the app off the public port so **everything** has
to come through HTTPS. Don't drop it.

---

## 5. Start everything

First run downloads several GB and takes 30–60 minutes.

```bash
cd ~/chatbot
SUPABASE_DOCKER_DIR=~/supabase/docker \
ADMIN_SERVICE_NUMBER=SUP001 \
ADMIN_NAME="Your Name" \
  bash deploy/airgap/bringup.sh --slim --no-load
```

It prints your **admin service number and a one-time password** — copy them now.

Then add the HTTPS front door:

```bash
cd ~/supabase/docker
docker compose --env-file .env \
  -f docker-compose.yml \
  -f ~/chatbot/deploy/airgap/docker-compose.airgap.yml \
  -f ~/chatbot/deploy/hosted/docker-compose.tls.yml \
  up -d caddy
```

Certificates are issued within a few seconds. Check:

```bash
curl -I https://sahayak.example.in/api/health     # expect HTTP/2 200
docker compose logs caddy | tail -20              # if it doesn't
```

Open `https://sahayak.example.in` — you should get the sign-in screen.

---

## 6. Load your documents

Put your PDFs in `~/kb` (from your laptop: `scp *.pdf user@server:~/kb/`), then:

```bash
cd ~/supabase/docker
docker compose --env-file .env \
  -f docker-compose.yml \
  -f ~/chatbot/deploy/airgap/docker-compose.airgap.yml \
  exec rag python cli.py ingest /kb --tier 1
```

Expect `OK <file>.pdf: N pages, N chunks`. Scanned pages are read with Hindi +
English OCR automatically; a large document takes a few minutes.

**Tier** decides who sees it: `1` = every jawan, `2`/`3` = progressively
restricted. Set each person's tier when you invite them.

You can also upload from **Admin → Documents** in the browser once you're signed in.

---

## 7. Sign in and invite people

1. Open `https://sahayak.example.in`, sign in with the admin credentials from Step 5.
2. Set a new password and enroll an authenticator app (Google Authenticator or
   similar). This is required — it's what protects the account.
3. **Admin → Users → + Invite user** for each jawan. Each gets a one-time
   password; they set their own and enroll their own authenticator on first login.
4. Send them the link. That's it — it works from any network, on any phone, and
   they can add it to their home screen.

---

## Answers come only from your documents

This is enforced, not requested:

- If the search doesn't find a good enough passage, the reply is a clean
  "not found" in Hindi and English — it never guesses.
- Every answer is checked before it's shown. If it doesn't cite a real passage
  from your documents, it's thrown away and replaced with the refusal.

**If a fair question gets refused**, that's the confidence dial, not a bug. Check
the document actually answers it *in words a jawan would type* — acronyms matter.
To adjust, change `RERANK_REFUSAL_THRESHOLD` in `.env` (default `0.35`; lower =
more answers but weaker grounding) and `docker compose ... up -d web`. Test with
[`../airgap/eval/`](../airgap/eval/) before and after.

---

## Day to day

| Task | Command |
|------|---------|
| Check it's up | `curl -s https://sahayak.example.in/api/health` |
| Logs | `docker compose ... logs -f web` |
| Add documents | Drop in `~/kb`, re-run Step 6 |
| Backup | `OUT_DIR=~/backups bash ~/chatbot/deploy/airgap/backup.sh` |
| Restart | Re-run Step 5 (safe — won't wipe data or re-create your admin) |

Full operations reference: [`../../RUNBOOK.md`](../../RUNBOOK.md).

**Set a nightly backup** — one line, and it's the difference between a bad day
and a lost knowledge base:

```bash
(crontab -l 2>/dev/null; echo "30 2 * * * OUT_DIR=$HOME/backups bash $HOME/chatbot/deploy/airgap/backup.sh") | crontab -
```

---

## Before you hand it over

- **Rotate the secrets** — see [`../../SECURITY.md`](../../SECURITY.md). Keys used
  during development shouldn't guard a public URL.
- **Keep the server patched** — `sudo apt update && sudo apt upgrade` on a schedule.
- **Restrict access further if asked** — set `IP_ALLOWLIST` in `.env` to limit the
  app to specific networks. Off by default, since jawans connect from anywhere.
