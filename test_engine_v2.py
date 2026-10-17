import json
from pathlib import Path
from engine import CashSecurityEngine

VIDEO_DIR = Path("/Users/almansurkakimov/Downloads/open/videos")
OUT = Path("baseline/step4_engine_test.json"); OUT.parent.mkdir(exist_ok=True)
EVID = Path("baseline/evidence"); EVID.mkdir(exist_ok=True)

CASES = [
    ("Х-11", "expect cashier_away"),
    ("Х-3",  "expect clean"),
    ("Х-26", "expect mobile_phone (has green pouch + phone)"),
]

eng = CashSecurityEngine(rules_csv="violation_rules_verified.csv")
results = {}
for name, expect in CASES:
    norm = name.replace("Х", "X").replace("х", "X").strip()
    f = next((v for v in VIDEO_DIR.glob("*.mov")
              if v.stem.replace("Х","X").replace("х","X").strip() == norm), None)
    if not f:
        print(f"[SKIP] {name} not found"); continue
    print(f"\n=== {name}  ({expect}) ===")
    r = eng.analyze_video(f, sample_fps=2, evidence_dir=str(EVID), dump_diag=True)
    csv_path = f"baseline/diag_{name.replace('Х','X')}.csv"
    eng.save_diagnostics_csv(csv_path)
    print(json.dumps(r, ensure_ascii=False, indent=2)[:2500])
    results[name] = r

OUT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
print(f"\nWrote {OUT}")
