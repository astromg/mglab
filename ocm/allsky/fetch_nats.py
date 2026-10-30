"""Download telescope zero points and raw-frame header cards from NATS JetStream into parquet files.

Read-only: an ordered (ephemeral) consumer over the retained `tic_status` stream (30 days). Writes
~/data/allsky/nats/zero_points.parquet (zero_monitor.lc of every telescope) and raw_frames.parquet (selected header
cards of tic.status.<tel>.fits.pipeline.raw, for the pointing ALT_TEL / AZ_TEL of each frame). The zero point of a
frame has oca_jd = JD - 2460000 of that frame.

    python fetch_nats.py
"""

import asyncio
import json
import os

import nats
import pandas as pd
from nats.js.api import ConsumerConfig, DeliverPolicy

NATS_URL = "nats://nats.oca.lan:4222"
OUT_DIR = os.path.expanduser("~/data/allsky/nats")
HEADER_CARDS = ["DATE-OBS", "JD", "TELESCOP", "OBSTYPE", "IMAGETYP", "SCIPROG", "OBJECT", "FILTER", "EXPTIME", "ALT_TEL", "AZ_TEL", "RA_TEL", "DEC_TEL", "AIRMASS"]


async def read_all(js, subject):
    """All retained messages of a (wildcard) subject: list of (subject, data dict)."""
    out = []
    sub = await js.subscribe(subject, ordered_consumer=True, config=ConsumerConfig(deliver_policy=DeliverPolicy.ALL))
    while True:
        try:
            msg = await sub.next_msg(timeout=10)
        except nats.errors.TimeoutError:
            break
        out.append((msg.subject, json.loads(msg.data)["data"]))
        if msg.metadata.num_pending == 0:
            break
    await sub.unsubscribe()
    return out


async def main():
    nc = await nats.connect(NATS_URL, connect_timeout=5)
    js = nc.jetstream()

    zero = await read_all(js, "tic.status.*.zero_monitor.lc")
    zp = pd.DataFrame([dict(telescope=s.split(".")[2], **d) for s, d in zero])
    zp = zp.drop_duplicates(["telescope", "oca_jd", "filter"])

    raw = await read_all(js, "tic.status.*.fits.pipeline.raw")
    rows = []
    for s, d in raw:
        h = d.get("raw", {}).get("header", {})
        rows.append(dict(telescope=s.split(".")[2], file_name=d.get("raw", {}).get("file_name"), raw_median=d.get("raw", {}).get("median"), **{c: h.get(c) for c in HEADER_CARDS}))
    frames = pd.DataFrame(rows)
    await nc.close()

    os.makedirs(OUT_DIR, exist_ok=True)
    zp.to_parquet(os.path.join(OUT_DIR, "zero_points.parquet"))
    frames.astype({c: str for c in ["DATE-OBS", "TELESCOP", "OBSTYPE", "IMAGETYP", "SCIPROG", "OBJECT", "FILTER"]}).to_parquet(os.path.join(OUT_DIR, "raw_frames.parquet"))
    print(f"{len(zp)} zero points, {len(frames)} raw frames -> {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
