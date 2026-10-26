# Demo video: voiceover script

`docs/demo.mp4` runs 68 seconds, phone format (780x1688), English captions burned in. It works silent. To add your voice, read this at a relaxed pace (about 160 words) and lay it under the video in any editor (CapCut, Clipchamp, iMovie). Sections follow the screens, not exact seconds.

| Screen | Say |
| --- | --- |
| Home | "When a dog goes missing in Israel, the posts are scattered across dozens of Facebook and WhatsApp groups. I built a Hebrew lost-and-found board that matches them automatically." |
| Report form | "A neighbor finds a dog. They add a photo, drop a pin and leave a phone number. No forms about breed or color: Claude reads the photo." |
| Matches | "Every open lost report is ranked by face similarity, using DINOv2 embeddings, plus distance and date, and each match explains itself in Hebrew." |
| Owner's page | "The owner sees the finder's phone. The finder never sees the owner's, which blocks the 'pay me to return your dog' scam." |
| Screenshot import | "Most posts live in group chats, so you can upload a screenshot. Claude fills in the report, including a phone number that right-to-left text displays backwards." |
| Results | "Everything is measured: the right dog comes first 86% of the time from the photo alone, and 96 to 98% with place and date. Built with Python, FastAPI and Claude." |

## Re-recording

```powershell
python scripts\seed_demo.py --dogs 300 --db demo.db
$env:PETMATCH_DB = "demo.db"; uvicorn app:app       # in one window, with ANTHROPIC_API_KEY set
python scripts\record_demo.py --db demo.db           # in another; needs ffmpeg and: pip install playwright; playwright install chromium
```

The 96-98% figure uses simulated places and dates (see the README); say "simulated" if anyone asks.
