"""Generate synthetic Hebrew lost/found group-post screenshots with an answer key.

Real screenshots carry real people's phone numbers, so the eval set is synthetic:
a generic group-post layout (no real platform branding), Hebrew text written the way
people actually post, a DogFaceNet photo for some dog posts, and made-up 555 phone numbers.

    python scripts/make_screenshots.py --n 40 --out eval/screenshots

Writes <out>/<id>.jpg and <out>/labels.jsonl (one answer key per screenshot).
Colors are only scored when the text states them (photo posts leave color to the photo).
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import date, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
]
BOLD_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
]

AREAS = [  # (city, full area text)
    ("רמת גן", "רמת גן, ליד הפארק הלאומי"), ("תל אביב", "תל אביב, פלורנטין"),
    ("ראש העין", "ראש העין, נווה אפק"), ("חיפה", "חיפה, הדר"), ("באר שבע", "באר שבע, שכונה ד'"),
    ("פתח תקווה", "פתח תקווה, כפר גנים"), ("ירושלים", "ירושלים, קטמונים"), ("רחובות", "רחובות, רחוב הרצל"),
    ("מודיעין", "מודיעין, מוריה"), ("כפר סבא", "כפר סבא, הירוקה"), ("נתניה", "נתניה, עיר ימים"),
    ("הרצליה", "הרצליה, נוף ים"), ("חולון", "חולון, קריית שרת"), ("אשדוד", "אשדוד, רובע ז'"),
]
COLOR_PHRASES = [  # (masculine, feminine) Hebrew phrase -> scored colors
    (("שחור", "שחורה"), ["black"]), (("לבן", "לבנה"), ["white"]), (("חום", "חומה"), ["brown"]),
    (("חום בהיר", "חומה בהירה"), ["tan"]), (("שחור לבן", "שחורה לבנה"), ["black", "white"]),
    (("זהוב", "זהובה"), ["golden"]), (("אפור", "אפורה"), ["gray"]),
    (("חום עם חזה לבן", "חומה עם חזה לבן"), ["brown", "white"]),
    (("שחור עם כתמים חומים", "שחורה עם כתמים חומים"), ["black", "brown"]),
]
CAT_COLOR_PHRASES = [
    (("ג'ינג'י", "ג'ינג'ית"), ["orange"]), (("אפור", "אפורה"), ["gray"]), (("שחור", "שחורה"), ["black"]),
    (("לבן עם כתמים שחורים", "לבנה עם כתמים שחורים"), ["white", "black"]),
    (("ג'ינג'י לבן", "ג'ינג'ית לבנה"), ["orange", "white"]),
]
EXTRAS = [("", ""), (" עם קולר אדום", " עם קולר אדום"), (" בלי קולר", " בלי קולר"),
          (" קטן וידידותי", " קטנה וידידותית"), (" קצת פחדן", " קצת פחדנית"),
          (" מאוד חברותי", " מאוד חברותית")]
DOG_NAMES = ["לוסי", "שוקו", "טופי", "רקסי", "מיקה", "בונבון", "לולה", "צ'ארלי", "בלו", "גינגר"]
CAT_NAMES = ["מיצי", "תום", "לונה", "שמנמן", "פיצי"]
POSTERS = ["נועה כ.", "אבי לוי", "מיכל ש.", "יוסי ב.", "דנה א.", "רון ג.", "שירה מ.", "עומר ד."]
GROUP = "אבודים ונמצאים - בעלי חיים"
MONTHS = ["ינואר", "פברואר", "מרץ", "אפריל", "מאי", "יוני", "יולי", "אוגוסט", "ספטמבר",
          "אוקטובר", "נובמבר", "דצמבר"]

W, PAD = 720, 28
INK, QUIET, BG, CARD = (20, 20, 20), (101, 103, 107), (240, 242, 245), (255, 255, 255)


def font(candidates, size):
    for c in candidates:
        if Path(c).exists():
            return ImageFont.truetype(c, size)
    raise FileNotFoundError("No Hebrew-capable font found; add one to FONT_CANDIDATES")


def wrap(text: str, f, width: int) -> list[str]:
    lines = []
    for para in text.split("\n"):
        words, line = para.split(" "), ""
        for w in words:
            trial = (line + " " + w).strip()
            if f.getlength(trial, direction="rtl", language="he") <= width:
                line = trial
            else:
                lines.append(line)
                line = w
        lines.append(line)
    return lines


def phone_variants(rng) -> tuple[str, str]:
    prefix = rng.choice(["050", "052", "053", "054", "058"])
    last = f"{rng.randint(0, 9999):04d}"
    digits = f"{prefix}555{last}"
    shown = rng.choice([f"{prefix}-555-{last}", f"{prefix} 555 {last}", digits, f"{prefix}-555{last}"])
    return digits, shown


def write_post(rng, post_date: date) -> tuple[str, dict]:
    species = rng.choices(["dog", "cat"], weights=[3, 1])[0]
    rtype = rng.choices(["lost", "found", "sighting"], weights=[5, 4, 1])[0]
    city, area = rng.choice(AREAS)
    with_photo = species == "dog" and rng.random() < 0.6

    days_ago, when = rng.choice([(0, "היום בבוקר"), (0, "הערב"), (1, "אתמול בלילה"), (1, "אתמול"),
                                 (2, "שלשום"), (None, None)])
    if days_ago is None:  # explicit date
        days_ago = rng.randint(2, 6)
        d = post_date - timedelta(days=days_ago)
        when = f"ב-{d.day}.{d.month}"
    event = post_date - timedelta(days=days_ago)

    female = rng.random() < 0.4
    g = 1 if female else 0  # index into (masculine, feminine) pairs

    phone_digits, phone_shown = phone_variants(rng)
    no_phone = rng.random() < 0.12
    contact = "פרטים בהודעה פרטית" if no_phone else rng.choice(
        [f"מי שראה {'אותה' if female else 'אותו'} בבקשה להתקשר {phone_shown}", f"טלפון: {phone_shown}",
         f"לפרטים {phone_shown}", f"נא ליצור קשר {phone_shown} תודה!!"])
    if with_photo:
        color_txt, colors = "", None  # color left to the photo, not scored
    else:
        phrases, colors = rng.choice(COLOR_PHRASES if species == "dog" else CAT_COLOR_PHRASES)
        color_txt = phrases[g]

    animal = {"dog": ("כלב", "כלבה"), "cat": ("חתול", "חתולה")}[species][g]
    name = None
    if rtype == "lost" and rng.random() < 0.8:
        name = rng.choice(DOG_NAMES if species == "dog" else CAT_NAMES)
    color_part = f" {color_txt}" if color_txt else ""
    extra = rng.choice(EXTRAS)[g]

    if rtype == "lost":
        title = rng.choice(["דחוף!", "עזרה!!", "אבד"])
        first = f"{title} {'נעלמה' if female else 'נעלם'} לנו {animal}{color_part}{extra}"
        if name:
            first += f", {'שמה' if female else 'שמו'} {name}"
        body = f"{first}.\n{'ברחה' if female else 'ברח'} {when} באזור {area}.\n{contact}"
    elif rtype == "found":
        first = rng.choice([f"{'נמצאה' if female else 'נמצא'} {animal}{color_part}{extra}",
                            f"מצאנו {animal}{color_part}{extra}"])
        body = (f"{first} {when} ב{area}.\n{'נמצאת' if female else 'נמצא'} אצלנו כרגע, "
                f"מחפשים את הבעלים.\n{contact}")
    else:
        reach = "פרטים בהודעה פרטית" if no_phone else f"אפשר להתקשר אליי {phone_shown}"
        body = (f"ראיתי {animal}{color_part}{extra} {'מסתובבת' if female else 'מסתובב'} לבד {when} "
                f"ב{area}.\nלא הצלחתי לתפוס. אם מישהו מחפש - {reach}")
    tag = "#כלבים" if species == "dog" else "#חתולים"
    body += "\n" + rng.choice(["שתפו בבקשה", f"#אבודים {tag}", "", "תודה לכולם"])

    label = {"report_type": rtype, "species": species, "pet_name": name, "colors": colors,
             "city": city, "area": area, "event_date": event.isoformat(),
             "contact_phone": None if no_phone else phone_digits, "has_pet_photo": with_photo}
    return body.strip(), label


def render(body: str, poster: str, post_date: date, photo: Path | None, out: Path) -> None:
    f_body, f_name, f_small = font(FONT_CANDIDATES, 25), font(BOLD_CANDIDATES, 24), font(FONT_CANDIDATES, 19)
    lines = wrap(body, f_body, W - 2 * PAD - 20)
    line_h = 38
    photo_h = 470 if photo else 0
    H = 150 + len(lines) * line_h + photo_h + 90
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([12, 12, W - 12, H - 12], radius=14, fill=CARD)
    # header: group, avatar, poster, date
    d.text((W - PAD, 34), GROUP, font=f_small, fill=QUIET, anchor="ra", direction="rtl", language="he")
    d.ellipse([W - PAD - 52, 66, W - PAD, 118], fill=(200, 210, 225))
    d.text((W - PAD - 64, 72), poster, font=f_name, fill=INK, anchor="ra", direction="rtl", language="he")
    stamp = f"{post_date.day} ב{MONTHS[post_date.month - 1]} {post_date.year} · {post_date.day % 12 + 8}:{(post_date.day * 7) % 60:02d}"
    d.text((W - PAD - 64, 104), stamp, font=f_small, fill=QUIET, anchor="ra", direction="rtl", language="he")
    y = 150
    for line in lines:
        d.text((W - PAD - 10, y), line, font=f_body, fill=INK, anchor="ra", direction="rtl", language="he")
        y += line_h
    if photo:
        with Image.open(photo) as p:
            pic = p.convert("RGB").resize((W - 2 * PAD, 450))
        im.paste(pic, (PAD, y + 10))
        y += photo_h
    d.line([PAD, y + 20, W - PAD, y + 20], fill=(222, 224, 228), width=2)
    for i, word in enumerate(["אהבתי", "תגובה", "שיתוף"]):
        d.text((W - PAD - 40 - i * 230, y + 38), word, font=f_small, fill=QUIET, anchor="ra",
               direction="rtl", language="he")
    im.save(out, "JPEG", quality=88)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", default="eval/screenshots")
    ap.add_argument("--photos", default="data/dogfacenet")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    photos = sorted(Path(args.photos).rglob("*.jpg"))
    if not photos:
        raise SystemExit(f"No photos under {args.photos}; run scripts/fetch_dogfacenet.py first")

    with (out / "labels.jsonl").open("w", encoding="utf-8") as f:
        for i in range(args.n):
            post_date = date(2026, 10, 1) + timedelta(days=rng.randint(0, 24))
            body, label = write_post(rng, post_date)
            photo = rng.choice(photos) if label["has_pet_photo"] else None
            sid = f"post_{i:03d}"
            render(body, rng.choice(POSTERS), post_date, photo, out / f"{sid}.jpg")
            label.update(id=sid, file=f"{sid}.jpg", post_date=post_date.isoformat(), text=body)
            f.write(json.dumps(label, ensure_ascii=False) + "\n")
    print(f"Wrote {args.n} screenshots + labels.jsonl to {out}")


if __name__ == "__main__":
    main()
