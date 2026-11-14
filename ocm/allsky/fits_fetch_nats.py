"""Read-only: allsky_overwatch clouds messages, telescope zero points and raw-frame pointing since START, from NATS
JetStream, into ~/data/allsky/fits/nats_*.parquet (for comparing the FITS study with the running service)."""
import asyncio, json, os, sys
from datetime import datetime, timezone
import nats
import pandas as pd
from nats.js.api import ConsumerConfig, DeliverPolicy

NATS_URL = "nats://nats.oca.lan:4222"
OUT = os.path.expanduser("~/data/allsky/fits")
START = datetime(2026, 11, 6, 0, 0, tzinfo=timezone.utc)
HEADER_CARDS = ["DATE-OBS", "JD", "IMAGETYP", "OBJECT", "FILTER", "EXPTIME", "ALT_TEL", "AZ_TEL", "AIRMASS"]


async def read_all(js, subject):
    out = []
    try:
        sub = await js.subscribe(subject, ordered_consumer=True, config=ConsumerConfig(
            deliver_policy=DeliverPolicy.BY_START_TIME, opt_start_time=START.isoformat().replace("+00:00", "Z")))
    except Exception as e:
        print("no stream for", subject, e); return out
    while True:
        try:
            msg = await sub.next_msg(timeout=10)
        except nats.errors.TimeoutError:
            break
        try:
            d = json.loads(msg.data)
            out.append((msg.subject, d.get("data", d), d.get("ts") or d.get("meta", {}).get("ts")))
        except Exception:
            pass
        if msg.metadata.num_pending == 0:
            break
    await sub.unsubscribe()
    print(subject, len(out))
    return out


def flat(d, prefix=""):
    r = {}
    for k, v in d.items():
        if isinstance(v, dict):
            r.update(flat(v, prefix + k + "."))
        elif not isinstance(v, list):
            r[prefix + k] = v
    return r


async def main():
    nc = await nats.connect(NATS_URL, connect_timeout=5)
    js = nc.jetstream()
    rows = []
    for subj in ("tic.status.ocm.allsky_overwatch.clouds", "data.overwatch.oca.allsky.clouds"):
        rows += [flat(d) for s, d, ts in await read_all(js, subj)]
    pd.DataFrame(rows).to_parquet(os.path.join(OUT, "nats_clouds.parquet"))
    zero = await read_all(js, "tic.status.*.zero_monitor.lc")
    zp = pd.DataFrame([dict(telescope=s.split(".")[2], **flat(d)) for s, d, ts in zero])
    zp.to_parquet(os.path.join(OUT, "nats_zero_points.parquet"))
    raw = await read_all(js, "tic.status.*.fits.pipeline.raw")
    fr = pd.DataFrame([dict(telescope=s.split(".")[2], file_name=d.get("raw", {}).get("file_name"),
                            **{c: str(d.get("raw", {}).get("header", {}).get(c)) for c in HEADER_CARDS}) for s, d, ts in raw])
    fr.to_parquet(os.path.join(OUT, "nats_raw_frames.parquet"))
    await nc.close()


asyncio.run(main())
