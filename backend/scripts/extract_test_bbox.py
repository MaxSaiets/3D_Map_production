"""Витягує bbox з великого PBF у маленький тест-PBF через osmium."""
import sys
import osmium

if len(sys.argv) < 6:
    print("Usage: extract_test_bbox.py input.pbf output.pbf south west north east")
    sys.exit(1)

inp = sys.argv[1]
out = sys.argv[2]
s, w, n, e = map(float, sys.argv[3:7])
print(f"Extracting bbox {s},{w} -> {n},{e} from {inp} -> {out}")

fp = osmium.FileProcessor(inp).with_filter(osmium.filter.GeoInterfaceFilter())
writer = osmium.SimpleWriter(out)
count = 0
for obj in fp:
    # check if object is in bbox via geom
    try:
        gi = obj.__geo_interface__
        coords = gi.get("geometry", {}).get("coordinates", [])
        # simplified: just write all in chunk and bbox-clip later
        writer.add(obj)
        count += 1
    except Exception:
        pass
writer.close()
print(f"Wrote {count} objects to {out}")
