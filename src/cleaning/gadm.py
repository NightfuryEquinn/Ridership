import json

# ── Load ──────────────────────────────────────────────────────────────────────
with open("data/gadm_mys_l1.json") as f:
    gadm = json.load(f)

features = gadm["features"]
print(f"=== GADM MYS Level-1 ===")
print(f"Raw feature count (States and Federal Territories): {len(features)}\n")
print(f"Raw feature properties: {features[0]['properties'].keys()}")
print(f"Raw feature properties count: {len(features[0]['properties'])}\n")

# Print the first five feature's properties
for i in range(5):
    print(f"Feature {i}: {features[i]['properties']}")

# ── 1. Drop always-empty columns (NL_NAME_1, CC_1) ───────────────────────────
# Both are 'NA' for all 16 states — no information content.
DROP_PROPS = {"NL_NAME_1", "CC_1"}

# ── 2. Sentinel 'NA' → None ───────────────────────────────────────────────────
# GADM uses the string 'NA' (not JSON null) for missing values.
# Convert to None so downstream tools treat them as actual nulls.

# ── 3. Normalise VARNAME_1 → list ─────────────────────────────────────────────
# VARNAME_1 stores pipe-delimited alternate names, e.g. "JohorDarulTakzim|Johore".
# Split into a list for easier querying; None where absent.

# ── 4. Validate geometry completeness ─────────────────────────────────────────
# All 16 features should have non-null MultiPolygon geometry.

cleaned_features = []
issues = []

for i, feat in enumerate(features):
    props = feat.get("properties", {})
    geom  = feat.get("geometry")
    gid   = props.get("GID_1", f"feature_{i}")

    # Geometry check
    if geom is None:
        issues.append(f"  [WARN] {gid}: null geometry — feature kept but flagged")
    elif geom.get("type") != "MultiPolygon":
        issues.append(f"  [WARN] {gid}: unexpected geometry type '{geom.get('type')}'")

    # Clean properties
    new_props = {}
    for k, v in props.items():
        if k in DROP_PROPS:
            continue                              # remove always-empty fields
        if v == "NA" or v == "":
            v = None                              # sentinel → null
        if k == "VARNAME_1" and v is not None:
            v = v.split("|")                      # pipe-list → Python list
        new_props[k] = v

    cleaned_features.append({**feat, "properties": new_props})

# ── 5. Duplicate GID_1 check ──────────────────────────────────────────────────
gids = [f["properties"]["GID_1"] for f in cleaned_features]
dup_gids = {g for g in gids if gids.count(g) > 1}
if dup_gids:
    issues.append(f"  [ERROR] Duplicate GID_1 values: {dup_gids}")

# ── 6. Expected state count check ─────────────────────────────────────────────
# Malaysia has 13 states + 3 federal territories = 16 administrative units.
if len(cleaned_features) != 16:
    issues.append(f"  [WARN] Expected 16 features, got {len(cleaned_features)}")

# ── 7. Report ─────────────────────────────────────────────────────────────────
print(f"\n=== GADM MYS Level-1 Cleaned ===")
print(f"Cleaned feature count: {len(cleaned_features)}")
print(f"Dropped property keys: {DROP_PROPS}")
print(f"Cleaned feature properties: {cleaned_features[0]['properties'].keys()}")
print(f"Cleaned feature properties count: {len(cleaned_features[0]['properties'])}\n")

if issues:
    print("Issues found:")
    for iss in issues:
        print(iss)
else:
    print("No issues found — all geometries present, no duplicate GIDs.")

# Spot-check a cleaned feature
sample = cleaned_features[0]["properties"]
print(f"\nSample cleaned properties (first feature):")
for k, v in sample.items():
    print(f"  {k}: {v!r}")

# ── 8. Export ─────────────────────────────────────────────────────────────────
gadm_clean = {**gadm, "features": cleaned_features}
with open("data/cleaned/gadm_mys_l1_clean.geojson", "w") as f:
    json.dump(gadm_clean, f, ensure_ascii=False, indent=2)
print("\nExported: data/cleaned/gadm_mys_l1_clean.geojson")