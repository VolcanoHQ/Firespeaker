#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Firespeaker user management (T2-1: the foundation).

Local-first identity for Volcano Studios: magic-link auth (links land in a dev
outbox file until real SMTP is configured), opaque session tokens in SQLite,
and the "local" -> real-user migration that the `owner` fields planted across
render jobs were waiting for.

Auth is OFF by default (FIRESPEAKER_AUTH unset/"off"): every surface behaves
exactly as before and ownership defaults to "local". With FIRESPEAKER_AUTH=on
the server refuses unauthenticated API access and stamps real user ids onto
everything owned.
"""

import argparse
import glob
import json
import logging
import os
import secrets
import sqlite3
import sys
import time
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger("UserDB")

DB_PATH = "data/users.db"
OUTBOX_DIR = "data/outbox"
SESSION_TTL_S = 30 * 24 * 3600      # 30 days
CODE_TTL_S = 15 * 60                # 15 minutes
COOKIE_NAME = "fs_session"


def auth_enabled() -> bool:
    return os.environ.get("FIRESPEAKER_AUTH", "off").lower() in ("on", "1", "true")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL,
        display_name TEXT, role TEXT DEFAULT 'owner', created_at REAL)""")
    # Profile migration: free-form profile document (roles list, bio, links).
    existing = {row[1] for row in c.execute("PRAGMA table_info(users);")}
    if "profile_json" not in existing:
        c.execute("ALTER TABLE users ADD COLUMN profile_json TEXT DEFAULT '{}';")
    c.execute("""CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY, user_id TEXT NOT NULL,
        created_at REAL, expires_at REAL)""")
    c.execute("""CREATE TABLE IF NOT EXISTS login_codes (
        code TEXT PRIMARY KEY, email TEXT NOT NULL,
        created_at REAL, expires_at REAL, used INTEGER DEFAULT 0)""")
    return c


def request_login(email: str) -> Optional[str]:
    """Issue a single-use login code. The 'email' lands in the dev outbox
    (data/outbox/) until real SMTP exists -- same flow, different transport."""
    email = (email or "").strip().lower()
    if not email or "@" not in email or len(email) > 254:
        return None
    code = secrets.token_urlsafe(24)
    now = time.time()
    c = _conn()
    c.execute("INSERT INTO login_codes (code, email, created_at, expires_at) VALUES (?,?,?,?)",
              (code, email, now, now + CODE_TTL_S))
    c.commit(); c.close()
    os.makedirs(OUTBOX_DIR, exist_ok=True)
    link = f"http://localhost:8082/api/auth/redeem?code={code}"
    with open(os.path.join(OUTBOX_DIR, f"login_{email.replace('@', '_at_')}.txt"), "w") as f:
        f.write(f"Volcano Studios sign-in link (valid {CODE_TTL_S // 60} min):\n{link}\n")
    logger.info(f"Login link for {email} written to dev outbox.")
    return code


def redeem_code(code: str) -> Optional[Dict[str, Any]]:
    """Single-use code -> session token (creating the user on first sign-in)."""
    if not code:
        return None
    now = time.time()
    c = _conn()
    row = c.execute("SELECT email, expires_at, used FROM login_codes WHERE code = ?", (code,)).fetchone()
    if not row or row[2] or row[1] < now:
        c.close()
        return None
    email = row[0]
    c.execute("UPDATE login_codes SET used = 1 WHERE code = ?", (code,))
    urow = c.execute("SELECT user_id FROM users WHERE email = ?", (email,)).fetchone()
    if urow:
        user_id = urow[0]
    else:
        user_id = "u_" + uuid.uuid4().hex[:12]
        c.execute("INSERT INTO users (user_id, email, display_name, created_at) VALUES (?,?,?,?)",
                  (user_id, email, email.split("@")[0], now))
        logger.info(f"Created user {user_id} for {email}.")
    token = secrets.token_urlsafe(32)
    c.execute("INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?,?,?,?)",
              (token, user_id, now, now + SESSION_TTL_S))
    c.commit(); c.close()
    return {"token": token, "user_id": user_id, "email": email}


def session_user(token: str) -> Optional[Dict[str, Any]]:
    if not token:
        return None
    c = _conn()
    row = c.execute("""SELECT u.user_id, u.email, u.display_name, u.role, s.expires_at
                       FROM sessions s JOIN users u ON u.user_id = s.user_id
                       WHERE s.token = ?""", (token,)).fetchone()
    c.close()
    if not row or row[4] < time.time():
        return None
    return {"user_id": row[0], "email": row[1], "display_name": row[2], "role": row[3]}


def logout(token: str) -> None:
    c = _conn()
    c.execute("DELETE FROM sessions WHERE token = ?", (token,))
    c.commit(); c.close()


VALID_ROLES = ("author", "voice_actor", "producer")


def get_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """User identity + profile document. Works for the implicit "local" user
    too (auth-off mode) so the account panel behaves identically in both modes."""
    if user_id == "local":
        return {"user_id": "local", "email": None, "display_name": "Local Studio",
                "roles": list(VALID_ROLES), "bio": "", "local": True}
    c = _conn()
    row = c.execute("SELECT email, display_name, profile_json FROM users WHERE user_id = ?",
                    (user_id,)).fetchone()
    c.close()
    if not row:
        return None
    profile = {}
    try:
        profile = json.loads(row[2] or "{}")
    except Exception:
        pass
    return {"user_id": user_id, "email": row[0], "display_name": row[1] or row[0].split("@")[0],
            "roles": profile.get("roles", ["author"]), "bio": profile.get("bio", ""),
            "local": False}


def update_profile(user_id: str, display_name: Optional[str] = None,
                   roles: Optional[list] = None, bio: Optional[str] = None) -> Optional[Dict[str, Any]]:
    if user_id == "local":
        return get_profile("local")  # the implicit user has no stored profile
    c = _conn()
    row = c.execute("SELECT profile_json FROM users WHERE user_id = ?", (user_id,)).fetchone()
    if not row:
        c.close()
        return None
    try:
        profile = json.loads(row[0] or "{}")
    except Exception:
        profile = {}
    if roles is not None:
        profile["roles"] = [r for r in roles if r in VALID_ROLES] or ["author"]
    if bio is not None:
        profile["bio"] = str(bio)[:1000]
    if display_name is not None and str(display_name).strip():
        c.execute("UPDATE users SET display_name = ? WHERE user_id = ?",
                  (str(display_name).strip()[:80], user_id))
    c.execute("UPDATE users SET profile_json = ? WHERE user_id = ?",
              (json.dumps(profile), user_id))
    c.commit(); c.close()
    return get_profile(user_id)


def claim_local(email: str) -> Dict[str, Any]:
    """Migration: assign every record still owned by "local" to the user with
    this email (creating them if needed). Covers every owner-bearing store;
    new stores must be added here as they appear."""
    email = (email or "").strip().lower()
    now = time.time()
    c = _conn()
    urow = c.execute("SELECT user_id FROM users WHERE email = ?", (email,)).fetchone()
    if urow:
        user_id = urow[0]
    else:
        user_id = "u_" + uuid.uuid4().hex[:12]
        c.execute("INSERT INTO users (user_id, email, display_name, created_at) VALUES (?,?,?,?)",
                  (user_id, email, email.split("@")[0], now))
    c.commit(); c.close()

    claimed = {"render_jobs": 0}
    for p in glob.glob("data/render_jobs/*.json"):
        try:
            with open(p, encoding="utf-8") as f:
                job = json.load(f)
            if job.get("owner") == "local":
                job["owner"] = user_id
                with open(p, "w", encoding="utf-8") as f:
                    json.dump(job, f, indent=2)
                claimed["render_jobs"] += 1
        except Exception as e:
            logger.warning(f"claim_local: skipping unreadable {p} ({e})")
    logger.info(f"Claimed for {email} ({user_id}): {claimed}")
    return {"user_id": user_id, "email": email, "claimed": claimed}


def main():
    p = argparse.ArgumentParser(description="Firespeaker user management")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("request-login"); s.add_argument("--email", required=True)
    s = sub.add_parser("claim-local"); s.add_argument("--email", required=True)
    s = sub.add_parser("list-users")
    a = p.parse_args()
    if a.cmd == "request-login":
        code = request_login(a.email)
        print(f"code issued (see {OUTBOX_DIR}/): {bool(code)}")
    elif a.cmd == "claim-local":
        print(json.dumps(claim_local(a.email), indent=2))
    elif a.cmd == "list-users":
        c = _conn()
        for r in c.execute("SELECT user_id, email, role, created_at FROM users"):
            print(r)
        c.close()


if __name__ == "__main__":
    main()
