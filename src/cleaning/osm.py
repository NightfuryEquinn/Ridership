import json, re
from collections import defaultdict  # used for stats counter


# ── Load ──────────────────────────────────────────────────────────────────────
with open("data/osm_pois.json") as f:
    osm = json.load(f)

elements = osm["elements"]
print(f"=== OSM POIs ===")
print(f"Raw element count: {len(elements)}")

# ── 1. Normalise coordinates ──────────────────────────────────────────────────
# nodes have direct lat/lon; ways have a 'center' dict from Overpass API.
# Promote center lat/lon to top-level for a uniform schema across both types.

def get_coords(elem):
    if elem["type"] == "node":
        return elem.get("lat"), elem.get("lon")
    elif elem["type"] == "way":
        center = elem.get("center", {})
        return center.get("lat"), center.get("lon")
    return None, None

# ── 2. Postcode validation and repair ─────────────────────────────────────────
# Malaysian postcodes are exactly 5 digits (e.g. 50450).
# The raw data contains 6-digit codes — these are Singapore-format postcodes
# that leaked in via cross-border OSM edits. Strip them.
MY_POSTCODE_RE = re.compile(r"^\d{5}$")

def clean_postcode(pc):
    if pc is None:
        return None
    pc = pc.strip()
    if MY_POSTCODE_RE.match(pc):
        return pc          # valid MY postcode
    return None            # 6-digit SG codes or other garbage → remove

# ── 3. Deduplicate by OSM ID ──────────────────────────────────────────────────
# All IDs in raw data are unique, but guard against future merges.
seen_ids = set()

# ── 4. Build cleaned element list ────────────────────────────────────────────
cleaned = []
stats = defaultdict(int)

for elem in elements:
    eid = elem["id"]
    if eid in seen_ids:
        stats["duplicate_id_skipped"] += 1
        continue
    seen_ids.add(eid)

    tags = elem.get("tags", {})
    lat, lon = get_coords(elem)

    # Flag elements that have no extractable coordinates (shouldn't happen)
    if lat is None or lon is None:
        stats["no_coords_skipped"] += 1
        continue

    # Validate postcode
    if "addr:postcode" in tags:
        cleaned_pc = clean_postcode(tags["addr:postcode"])
        if cleaned_pc is None:
            stats["bad_postcode_removed"] += 1
            tags = {k: v for k, v in tags.items() if k != "addr:postcode"}
        else:
            tags["addr:postcode"] = cleaned_pc

    # Build unified element
    new_elem = {
        "id":   eid,
        "type": elem["type"],
        "lat":  lat,
        "lon":  lon,
        "tags": tags,
    }
    cleaned.append(new_elem)
    stats["kept"] += 1

# ── 5. Flag name-completeness ─────────────────────────────────────────────────
no_name = sum(1 for e in cleaned if "name" not in e["tags"])
stats["no_name_tag"] = no_name

# Print the first five elements without a name tag
for i in range(240, 250):
  e = cleaned[i]
  if "name" not in e["tags"]:
    print(f"Element {i} has no name tag: {e['tags']}")

# ── 6. Report ────────────────────────────────────────────────────────────────
print(f"\nCleaning summary:")
print(f"  Kept:                        {stats['kept']:,}")
print(f"  Duplicate IDs skipped:       {stats['duplicate_id_skipped']:,}")
print(f"  No-coordinate skipped:       {stats['no_coords_skipped']:,}")
print(f"  Malformed postcodes removed: {stats['bad_postcode_removed']:,}")
print(f"  Elements without name tag:   {stats['no_name_tag']:,} ({100*stats['no_name_tag']/stats['kept']:.1f}%)")

# ── 7. Export ─────────────────────────────────────────────────────────────────
out = {**{k: v for k, v in osm.items() if k != "elements"}, "elements": cleaned}
with open("data/cleaned/osm_pois_clean.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print("\nExported: data/cleaned/osm_pois_clean.json")