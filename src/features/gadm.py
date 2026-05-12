"""
gadm.py  — Cleaning + ST-model preparation
Changes vs original:
  - Added state centroid extraction (lat/lon) for distance-based edge weights.
  - Added shared-border adjacency matrix using shapely geometry intersection.
    This is required by GCN-SBULSTM / TGACN / STGNN-STEP when modelling
    state-level features (rainfall, ridership aggregated to state level).
  - Adjacency matrix exported as:
      (a) adjacency_matrix.npy   — N×N binary numpy array
      (b) adjacency_edges.csv    — edge list with shared_border_length_km
      (c) state_nodes.csv        — node index → GID_1 / state name / centroid
"""

import json
import numpy as np
import pandas as pd

# shapely is needed for geometry operations; it ships with geopandas
try:
    from shapely.geometry import shape, mapping
    from shapely.ops import unary_union
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    print("[WARN] shapely not installed — skipping adjacency matrix. "
          "Run: pip install shapely")

TOUCH_THRESHOLD = 1e-6   # degrees; shared segments shorter than this = water/point only


def main():
    # ── Load ──────────────────────────────────────────────────────────────────────
    with open("data/raw/gadm_mys_l1.json") as f:
        gadm = json.load(f)

    features = gadm["features"]
    print(f"=== GADM MYS Level-1 ===")
    print(f"Raw feature count: {len(features)}")

    # ── 1–8. Original cleaning (unchanged) ───────────────────────────────────────
    DROP_PROPS = {"NL_NAME_1", "CC_1"}

    cleaned_features = []
    issues = []

    for i, feat in enumerate(features):
        props = feat.get("properties", {})
        geom  = feat.get("geometry")
        gid   = props.get("GID_1", f"feature_{i}")

        if geom is None:
            issues.append(f"  [WARN] {gid}: null geometry — feature kept but flagged")
        elif geom.get("type") not in ("MultiPolygon", "Polygon"):
            issues.append(f"  [WARN] {gid}: unexpected geometry type '{geom.get('type')}'")

        new_props = {}
        for k, v in props.items():
            if k in DROP_PROPS:
                continue
            if v == "NA" or v == "":
                v = None
            if k == "VARNAME_1" and v is not None:
                v = v.split("|")
            new_props[k] = v

        cleaned_features.append({**feat, "properties": new_props})

    gids = [f["properties"]["GID_1"] for f in cleaned_features]
    dup_gids = {g for g in gids if gids.count(g) > 1}
    if dup_gids:
        issues.append(f"  [ERROR] Duplicate GID_1: {dup_gids}")
    if len(cleaned_features) != 16:
        issues.append(f"  [WARN] Expected 16 features, got {len(cleaned_features)}")

    print(f"\nCleaned feature count: {len(cleaned_features)}")
    if issues:
        for iss in issues:
            print(iss)
    else:
        print("No issues found.")

    gadm_clean = {**gadm, "features": cleaned_features}
    with open("data/cleaned/gadm_mys_l1_clean.geojson", "w") as f:
        json.dump(gadm_clean, f, ensure_ascii=False, indent=2)
    print("Exported: data/cleaned/gadm_mys_l1_clean.geojson")

    # ── NEW: Centroid extraction ──────────────────────────────────────────────────
    node_records = []
    geom_shapes  = {}   # gid → shapely geometry (used for adjacency below)

    for feat in cleaned_features:
        gid   = feat["properties"]["GID_1"]
        name  = feat["properties"].get("NAME_1", gid)
        geom  = feat.get("geometry")

        if geom is None or not SHAPELY_AVAILABLE:
            centroid_lat, centroid_lon = None, None
            geom_shape = None
        else:
            try:
                shp = shape(geom)
                centroid_lon = shp.centroid.x
                centroid_lat = shp.centroid.y
                geom_shape   = shp
            except Exception as e:
                print(f"  [WARN] {gid}: centroid failed — {e}")
                centroid_lat, centroid_lon = None, None
                geom_shape = None

        geom_shapes[gid] = geom_shape
        node_records.append({
            "node_idx":     len(node_records),
            "gid_1":        gid,
            "name":         name,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
        })

    df_nodes = pd.DataFrame(node_records)
    print(f"\nNode table shape: {df_nodes.shape}")
    print(df_nodes[["node_idx", "gid_1", "name", "centroid_lat", "centroid_lon"]].to_string(index=False))

    df_nodes.to_csv("data/cleaned/gadm_state_nodes.csv", index=False)
    print("Exported: data/cleaned/gadm_state_nodes.csv")

    # ── NEW: Shared-border adjacency matrix ───────────────────────────────────────
    if SHAPELY_AVAILABLE:
        n = len(df_nodes)
        gid_list = df_nodes["gid_1"].tolist()
        A = np.zeros((n, n), dtype=np.float32)
        edges = []

        for i, gid_i in enumerate(gid_list):
            for j, gid_j in enumerate(gid_list):
                if i >= j:
                    continue
                shp_i = geom_shapes.get(gid_i)
                shp_j = geom_shapes.get(gid_j)
                if shp_i is None or shp_j is None:
                    continue
                try:
                    intersection = shp_i.boundary.intersection(shp_j.boundary)
                    shared_len   = intersection.length   # degrees
                except Exception:
                    shared_len = 0.0

                if shared_len > TOUCH_THRESHOLD:
                    shared_km = shared_len * 111.0
                    A[i, j] = shared_km
                    A[j, i] = shared_km
                    edges.append({
                        "node_i":   i,
                        "node_j":   j,
                        "gid_i":    gid_i,
                        "gid_j":    gid_j,
                        "shared_border_km": round(shared_km, 2),
                    })

        A_binary = (A > 0).astype(np.float32)

        print(f"\nAdjacency matrix — {A_binary.sum().astype(int) // 2} state-pair borders detected")
        np.save("data/cleaned/gadm_adj_matrix.npy",        A_binary)
        np.save("data/cleaned/gadm_adj_matrix_weighted.npy", A)

        df_edges = pd.DataFrame(edges)
        df_edges.to_csv("data/cleaned/gadm_adj_edges.csv", index=False)

        print("Exported:")
        print("  data/cleaned/gadm_adj_matrix.npy           (N×N binary)")
        print("  data/cleaned/gadm_adj_matrix_weighted.npy  (N×N shared-border-km weights)")
        print("  data/cleaned/gadm_adj_edges.csv            (edge list with weights)")
        print("\nTo add self-loops for standard GCN: A_hat = A_binary + np.eye(N)")
    else:
        print("\n[SKIP] Adjacency matrix not built — install shapely and re-run.")


if __name__ == "__main__":
    main()
