"""Email alerts. ConsoleNotifier for development, ResendNotifier for real email.

Privacy rule: the lost-pet owner receives the finder's contact details (a finder
wants to be reached). The finder never receives the owner's - that protects owners
from fake "I found your dog, pay for delivery" messages.
"""
from __future__ import annotations

import html
import json
import os
import urllib.request

from .ranking import HE_COLORS


class Notifier:
    def send(self, to: str, subject: str, html_body: str) -> None:
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    """Keeps sent emails in memory and prints them - for development and tests."""
    def __init__(self, quiet: bool = False) -> None:
        self.sent: list[dict] = []
        self.quiet = quiet

    def send(self, to, subject, html_body):
        self.sent.append({"to": to, "subject": subject, "html": html_body})
        if not self.quiet:
            print(f"[email] to={to} subject={subject}")


class ResendNotifier(Notifier):
    """https://resend.com - free tier: 3,000 emails/month, 100/day."""
    def __init__(self, api_key: str | None = None, sender: str | None = None) -> None:
        self.api_key = api_key or os.environ["RESEND_API_KEY"]
        self.sender = sender or os.environ.get("ALERT_FROM", "Pet Matcher <onboarding@resend.dev>")

    def send(self, to, subject, html_body):
        body = json.dumps({"from": self.sender, "to": [to], "subject": subject, "html": html_body}).encode()
        req = urllib.request.Request("https://api.resend.com/emails", data=body, method="POST",
                                     headers={"Authorization": f"Bearer {self.api_key}",
                                              "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()


def match_email(lost, found, reason: str, link: str) -> tuple[str, str]:
    """Hebrew, right-to-left email to the owner of the lost report."""
    name = lost.pet_name or "חיית המחמד שלך"
    e = html.escape
    colors = ", ".join(HE_COLORS.get(c, c) for c in found.colors) or "לא צוין"
    contact = " · ".join(e(x) for x in (found.contact_name, found.contact_phone, found.contact_email) if x)
    subject = f"ייתכן שמצאו את {name}"
    body = f"""<div dir="rtl" style="font-family:Arial,sans-serif;font-size:16px;line-height:1.6">
<h2 style="margin:0 0 12px">ייתכן שמצאו את {e(name)}</h2>
<p>דיווח חדש על חיה שנמצאה דומה מאוד לדיווח שלך.</p>
<p><b>למה זה נראה כמו התאמה:</b> {e(reason)}</p>
<p><b>צבע:</b> {e(colors)} · <b>אזור:</b> {e(found.area_text or "לא צוין")} · <b>תאריך:</b> {found.event_date:%d.%m.%Y}</p>
<p><b>יצירת קשר עם המוצא/ת:</b> {contact}</p>
<p><a href="{e(link)}">לכל ההתאמות האפשריות</a></p>
<p style="color:#666;font-size:13px">חשוב: אל תשלמו לאף אחד מראש. בעלים אמיתיים יכולים להוכיח בעלות
בתמונות, שבב או וטרינר. ההתאמה נעשתה אוטומטית ועלולה לטעות.</p>
</div>"""
    return subject, body
