# Free forever — Oracle Cloud + DuckDNS

End-to-end setup on Oracle's **Always Free** tier: 4 CPU cores, 24 GB RAM, and a
free HTTPS address. Total cost **₹0/month**, permanently — not a trial.

That's enough to run everything on one machine, including the language model, so
there are no per-question charges either.

**Time:** about 2 hours, most of it waiting for downloads.

> **Two Oracle-specific traps** are covered below and catch nearly everyone:
> the **two-layer firewall** (Part 4) and **capacity errors** (Part 1). Don't skip
> those sections.

---

## Part 1 — Create the server

1. Sign up at **cloud.oracle.com** (needs a card for identity check — Always Free
   resources are never charged).
2. **Menu → Compute → Instances → Create instance**
3. Set these:

   | Field | Value |
   |---|---|
   | Image | **Ubuntu 24.04** |
   | Shape | **VM.Standard.A1.Flex** ← the free ARM one |
   | OCPUs | **4** |
   | Memory | **24 GB** |
   | Boot volume | **100 GB** (free allowance is 200 GB) |

4. Under **Add SSH keys**, choose *Generate a key pair* and **download the private
   key**. You cannot get it again.
5. Click **Create**, then copy the **Public IP address**.

> ### 🔴 "Out of host capacity"
> Common on the free ARM shape — it means that region is full right now, not that
> you did anything wrong. Fixes, in order:
> 1. Change **Availability Domain** (AD-1 → AD-2 → AD-3) and retry.
> 2. Retry every few hours — capacity frees up constantly.
> 3. If your home region stays full, create a new account in a quieter region.
>
> Most people get in within a day. Keep trying; it's worth ₹0/month.

---

## Part 2 — Get a free web address

Oracle gives you an IP, but HTTPS needs a name. **DuckDNS** gives them free.

1. Go to **duckdns.org**, sign in with Google/GitHub.
2. Create **two** subdomains — you need both, the second is what the login screen
   talks to:
   - `sainik` → becomes `sainik.duckdns.org`
   - `sainikapi` → becomes `sainikapi.duckdns.org`
3. Put your Oracle **public IP** in the *current ip* box for each and click
   **update ip**.

Check from your own machine:

```bash
ping sainik.duckdns.org
ping sainikapi.duckdns.org
```

✅ Both must show your server's IP before you continue.

---

## Part 3 — Connect to the server

```bash
chmod 600 ~/Downloads/ssh-key-*.key          # the file you downloaded
ssh -i ~/Downloads/ssh-key-*.key ubuntu@YOUR.SERVER.IP
```

The username is **`ubuntu`**, not root.

*(On Windows, use PowerShell — it has `ssh` built in.)*

---

## Part 4 — Open the firewall (both layers) 🔴

**This is the step everyone misses.** Oracle blocks traffic in *two* independent
places. Open only one and the site silently never loads.

### Layer 1 — in the Oracle website

**Instance → Virtual Cloud Network → Security Lists → Default Security List →
Add Ingress Rules**, and add two:

| Source CIDR | Protocol | Destination Port |
|---|---|---|
| `0.0.0.0/0` | TCP | `80` |
| `0.0.0.0/0` | TCP | `443` |

### Layer 2 — on the server itself

Ubuntu images from Oracle ship with a firewall that blocks everything but SSH:

```bash
sudo iptables -I INPUT 1 -p tcp --dport 80 -j ACCEPT
sudo iptables -I INPUT 1 -p tcp --dport 443 -j ACCEPT
sudo netfilter-persistent save
```

✅ Check both worked: `sudo iptables -L INPUT -n --line-numbers | head` should
show your ACCEPT rules for 80 and 443 at the top.

---

## Part 5 — Install Docker and the code

```bash
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && exec su -l $USER

cd ~
git clone https://github.com/mannpandya1702/chatbot.git
cd chatbot && git checkout claude/repository-review-5ox08x && cd ~
git clone --depth 1 https://github.com/supabase/supabase.git
mkdir -p ~/kb
```

✅ `docker version` prints a version.

---

## Part 6 — Generate all passwords and keys

Substitute your own DuckDNS names:

```bash
cd ~/chatbot
SUPABASE_DOCKER_DIR=~/supabase/docker \
SUPABASE_PUBLIC_URL=https://sainikapi.duckdns.org \
GEN_MODEL=qwen2.5:7b-instruct \
KB_DIR=$HOME/kb \
  bash deploy/airgap/bootstrap-env.sh
```

Every password and key is generated for you — you never invent one.

---

## Part 7 — Add the HTTPS settings

```bash
cd ~/supabase/docker
cat >> .env <<EOF

APP_DOMAIN=sainik.duckdns.org
API_DOMAIN=sainikapi.duckdns.org
SITE_URL=https://sainik.duckdns.org
CADDYFILE=$HOME/chatbot/deploy/hosted/Caddyfile
WEB_PORT=127.0.0.1:3000
EOF
```

Then `nano .env`, scroll to the bottom, and put in **your** DuckDNS names. Leave
the last two lines exactly as written.

---

## Part 8 — Start everything ☕

First run builds the images and downloads several GB. On 4 ARM cores expect
**45–75 minutes**. Grab lunch.

```bash
cd ~/chatbot
SUPABASE_DOCKER_DIR=~/supabase/docker \
ADMIN_SERVICE_NUMBER=SUP001 \
ADMIN_NAME="Your Name" \
  bash deploy/airgap/bringup.sh --slim --no-load
```

✅ Ends with **"Sainik Sahayak is up"** and prints your admin service number and a
**one-time password**.

### 📋 Copy that password now — it is shown once.

---

## Part 9 — Switch on HTTPS

```bash
cd ~/supabase/docker
docker compose --env-file .env \
  -f docker-compose.yml \
  -f ~/chatbot/deploy/airgap/docker-compose.airgap.yml \
  -f ~/chatbot/deploy/hosted/docker-compose.tls.yml \
  up -d caddy
```

Wait 30 seconds, then:

```bash
curl -I https://sainik.duckdns.org/api/health
```

✅ `HTTP/2 200` — you are live on the internet with a real certificate.

If not: `docker compose logs caddy | tail -30`. Nine times out of ten it's Part 4.

---

## Part 10 — Keep Oracle from reclaiming it

Oracle reclaims Always Free machines that look idle. One line of insurance:

```bash
(crontab -l 2>/dev/null; echo "*/10 * * * * curl -s https://sainik.duckdns.org/api/health >/dev/null") | crontab -
```

While you're there, add the nightly backup:

```bash
(crontab -l 2>/dev/null; echo "30 2 * * * OUT_DIR=\$HOME/backups bash \$HOME/chatbot/deploy/airgap/backup.sh") | crontab -
```

---

## Part 11 — Sign in and load your documents

1. Open **`https://sainik.duckdns.org`**
2. Sign in with the service number and one-time password from Part 8
3. Set a new password, then scan the QR code with **Google Authenticator**
4. **Admin → Documents → Upload** — pick a PDF, choose who can see it
   (tier 1 = everyone), upload. Scanned pages are OCR'd in Hindi and English
   automatically; status flips to **ready** when it's searchable.
5. **Chat** — ask something your document answers. You should get an answer with
   its source underneath. Ask something unrelated; you should get a clean
   *not found*. That is the system working correctly.
6. **Admin → Users → + Invite user** for each person, then send them the link.

Done. It works from any phone, on any network, anywhere — for ₹0/month.

---

## If something breaks

| Symptom | Look here |
|---|---|
| Site won't load at all | **Part 4** — almost always the second firewall layer |
| `curl` returns nothing | `docker compose logs caddy \| tail -30` |
| Certificate error | DNS not pointing at the server yet — re-check Part 2 |
| Bring-up stops | Send the last 20 lines it printed |
| Image build fails | ARM wheel issue — send the failing `pip` line |
| Slow answers | Normal on 4 CPU cores. Swap to `qwen2.5:3b-instruct` in `.env` and `docker compose ... up -d web` |

Day-to-day operations: [`../../RUNBOOK.md`](../../RUNBOOK.md).
