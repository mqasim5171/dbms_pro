import os, random
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv
from pymongo import MongoClient

# ✅ Load backend .env
ROOT_DIR = Path(__file__).resolve().parents[1]   # dbms_pro/
ENV_PATH = ROOT_DIR / "backend" / ".env"
load_dotenv(ENV_PATH)

MONGO_URL = os.getenv("MONGO_URL")
DB_NAME = os.getenv("DB_NAME")

if not MONGO_URL or not DB_NAME:
    raise SystemExit(f"❌ Missing MONGO_URL/DB_NAME. Checked: {ENV_PATH}")

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def bucket_price(price: float) -> str:
    if price < 150000: return "low"
    if price < 500000: return "mid"
    return "high"

def bucket_beds(b: int) -> str:
    if b <= 2: return "1-2"
    if b <= 4: return "3-4"
    return "5+"

def most_common(values, default="unknown"):
    if not values: return default
    return Counter(values).most_common(1)[0][0]

def main():
    client = MongoClient(MONGO_URL, serverSelectionTimeoutMS=8000)
    db = client[DB_NAME]
    db.command("ping")

    users = list(db.users.find({}, {"_id": 0, "id": 1, "role": 1}))
    props = list(db.properties.find({}, {"_id": 0, "id": 1, "city_id": 1, "property_type_id": 1, "price": 1, "bedrooms": 1}))
    favs = list(db.favorites.find({}, {"_id": 0, "user_id": 1, "property_id": 1}))
    books = list(db.bookings.find({}, {"_id": 0, "user_id": 1, "property_id": 1}))

    if not users or not props:
        raise SystemExit("❌ users/properties empty. Run /api/admin/seed then /api/admin/seed-reco first.")
    if not favs and not books:
        raise SystemExit("❌ No favorites/bookings found. Run /api/admin/seed-reco first.")

    # liked pairs
    liked_pairs = set()
    for x in favs:
        liked_pairs.add((x["user_id"], x["property_id"]))
    for x in books:
        liked_pairs.add((x["user_id"], x["property_id"]))

    prop_map = {p["id"]: p for p in props}

    # build preferences from liked history
    hist_city = defaultdict(list)
    hist_type = defaultdict(list)
    hist_price = defaultdict(list)

    for (uid, pid) in liked_pairs:
        p = prop_map.get(pid)
        if not p:
            continue
        hist_city[uid].append(str(p.get("city_id", "unknown")))
        hist_type[uid].append(str(p.get("property_type_id", "unknown")))
        hist_price[uid].append(bucket_price(float(p.get("price", 0))))

    buyers = [u for u in users if str(u.get("role", "")).lower() == "buyer"]
    if not buyers:
        raise SystemExit("❌ No buyer users found.")

    random.seed(42)
    rows = []

    # ✅ per buyer: include all positives we can + matched count of hard negatives
    max_pos_per_user = 10
    neg_ratio = 3  # negatives per positive

    for u in buyers:
        uid = u["id"]
        pref_city = most_common(hist_city[uid])
        pref_type = most_common(hist_type[uid])
        pref_price = most_common(hist_price[uid])

        # collect this user's positives
        user_pos = [pid for (xuid, pid) in liked_pairs if xuid == uid and pid in prop_map]
        random.shuffle(user_pos)
        user_pos = user_pos[:max_pos_per_user]

        if not user_pos:
            # no history => skip user (or we can add cold-start rows, but better skip)
            continue

        # for each positive, add hard negatives (mismatched)
        for pid in user_pos:
            p = prop_map[pid]

            # positive row
            rows.append({
                "pref_city": pref_city,
                "pref_type": pref_type,
                "pref_price_bucket": pref_price,
                "prop_city": str(p.get("city_id", "unknown")),
                "prop_type": str(p.get("property_type_id", "unknown")),
                "price_bucket": bucket_price(float(p.get("price", 0))),
                "beds_bucket": bucket_beds(int(p.get("bedrooms", 0))),
                "liked": 1,
            })

            # hard negatives: mismatch (different city OR different type OR different price bucket)
            hard_candidates = []
            for q in props:
                if q["id"] == pid:
                    continue
                q_city = str(q.get("city_id", "unknown"))
                q_type = str(q.get("property_type_id", "unknown"))
                q_price = bucket_price(float(q.get("price", 0)))

                mismatch = (
                    (pref_city != "unknown" and q_city != pref_city) or
                    (pref_type != "unknown" and q_type != pref_type) or
                    (pref_price != "unknown" and q_price != pref_price)
                )
                if mismatch:
                    hard_candidates.append(q)

            if not hard_candidates:
                continue

            negs = random.sample(hard_candidates, k=min(len(hard_candidates), neg_ratio))
            for q in negs:
                rows.append({
                    "pref_city": pref_city,
                    "pref_type": pref_type,
                    "pref_price_bucket": pref_price,
                    "prop_city": str(q.get("city_id", "unknown")),
                    "prop_type": str(q.get("property_type_id", "unknown")),
                    "price_bucket": bucket_price(float(q.get("price", 0))),
                    "beds_bucket": bucket_beds(int(q.get("bedrooms", 0))),
                    "liked": 0,
                })

    df = pd.DataFrame(rows)

    if df.empty:
        raise SystemExit("❌ Export produced no rows. Increase interactions by re-running seed-reco.")

    out_path = os.path.join(OUT_DIR, "reco_dataset_hardneg.csv")
    df.to_csv(out_path, index=False)

    print("✅ Connected to:", MONGO_URL)
    print("✅ DB:", DB_NAME)
    print("✅ Saved:", out_path)
    print("Rows:", df.shape[0])
    print("Liked counts:\n", df["liked"].value_counts())

if __name__ == "__main__":
    main()
