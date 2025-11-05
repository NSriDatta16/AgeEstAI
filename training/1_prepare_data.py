import os, re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

AGE_BINS = [(0,12),"0-12",(13,19),"13-19",(20,29),"20-29",(30,39),"30-39",(40,49),"40-49",(50,64),"50-64",(65,200),"65+"]
EMO7 = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def to_age_bin(age:int)->str:
    for i in range(0,len(AGE_BINS),2):
        lo, hi = AGE_BINS[i]; name = AGE_BINS[i+1]
        if lo <= age <= hi: return name
    return "20-29"

def norm_gender(x):
    if isinstance(x,str):
        x=x.strip().lower()
        if x in ["m","male","1","man","boy"]: return "male"
        if x in ["f","female","0","woman","girl"]: return "female"
    if isinstance(x,(int,float)):
        return "male" if int(x)==1 else "female"
    return None

# ---------- UTKFace ----------
def load_utkface():
    rows=[]
    utk_dir = RAW/"utkface"
    pat = re.compile(r"^(\d+)_(\d+)_")
    for p in utk_dir.glob("*.jpg*"):
        m = pat.match(p.name)
        if not m: continue
        age = int(m.group(1)); gender = norm_gender(int(m.group(2)))
        if gender is None: continue
        rows.append({"image_path":str(p.resolve()), "age_bin":to_age_bin(age), "gender":gender})
    return pd.DataFrame(rows)

# ---------- FairFace (robust) ----------
def load_fairface():
    base = RAW / "fairface"
    rows = []

    # Accept common CSV names
    candidates = [
        ("train_labels.csv", "train"),
        ("val_labels.csv", "val"),
        ("fairface_label_train.csv", "train"),
        ("fairface_label_val.csv", "val"),
        ("label_train.csv", "train"),
        ("label_val.csv", "val"),
    ]

    def fix_age_bin(s):
        s = str(s).replace(" ", "").lower()
        mapping = {
            "0-2": "0-12",
            "3-9": "0-12",
            "10-19": "13-19",
            "20-29": "20-29",
            "30-39": "30-39",
            "40-49": "40-49",
            "50-59": "50-64",
            "60-69": "50-64",
            "morethan70": "65+",
            "70+": "65+",
            ">70": "65+",
            "morethan70years": "65+",
        }
        return mapping.get(s, "20-29")

    # Build a fast filename -> fullpath index once per split
    def build_index(split_dir: Path):
        idx = {}
        # FairFace usually stores images directly under split without subfolders
        for p in split_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                idx[p.name] = p.resolve()
        return idx

    for csv_name, split in candidates:
        csv_path = base / csv_name
        if not csv_path.exists():
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            print(f"Skipping FairFace {csv_name} (unreadable)")
            continue

        # figure columns
        file_col = next((c for c in ["file", "path", "img", "image_file", "image", "filename"] if c in df.columns), None)
        gender_col = next((c for c in ["gender", "Gender"] if c in df.columns), None)
        age_col = next((c for c in ["age", "Age", "age_group"] if c in df.columns), None)
        if not (file_col and gender_col and age_col):
            print(f"Skipping FairFace {csv_name} (missing expected columns)")
            continue

        split_dir = base / split
        if not split_dir.exists():
            print(f"Skipping FairFace {csv_name} (split dir missing: {split_dir})")
            continue

        # Build index once
        index = build_index(split_dir)

        # Map rows quickly via the index
        for _, r in df.iterrows():
            rel = r[file_col]
            if pd.isna(rel):
                continue
            fname = Path(str(rel)).name
            img_path = index.get(fname)
            if img_path is None:
                # sometimes CSV already has full relative path; try direct
                candidate = (split_dir / str(rel)).resolve()
                if candidate.exists():
                    img_path = candidate
                else:
                    continue

            gender = norm_gender(r[gender_col])
            age_bin = fix_age_bin(r[age_col])
            if gender is None:
                continue
            rows.append({"image_path": str(img_path), "age_bin": age_bin, "gender": gender})

    return pd.DataFrame(rows)


# ---------- RAF-DB (very robust) ----------
def load_rafdb():
    base = RAW/"rafdb"
    rows=[]
    # If CSVs exist, use them—but resolve paths via rglob
    for split in ["train","test"]:
        csv_path = base/f"{split}_labels.csv"
        split_dir = base/split
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                print(f"Skipping RAF-DB {csv_path.name} (unreadable/empty)")
                df = pd.DataFrame()

            if not df.empty:
                file_col = next((c for c in ["file","name","path","img","filename"] if c in df.columns), None)
                emo_col = next((c for c in ["emotion","label","class"] if c in df.columns), None)
                if file_col and emo_col:
                    for _,r in df.iterrows():
                        fname = str(r[file_col])
                        if pd.isna(fname): 
                            continue
                        # find image anywhere under split
                        found = list(split_dir.rglob(Path(fname).name))
                        if not found:
                            continue
                        emo = str(r[emo_col]).strip().lower()
                        # map digits to emo name if needed
                        try:
                            emo_i = int(emo)
                            emo = {1:"surprise",2:"fear",3:"disgust",4:"happy",5:"sad",6:"angry",7:"neutral"}.get(emo_i, None)
                        except:
                            pass
                        # normalize
                        map2 = {"happiness":"happy","surprised":"surprise","anger":"angry","sadness":"sad"}
                        emo = map2.get(emo, emo)
                        if emo in EMO7:
                            rows.append({"image_path":str(found[0].resolve()), "emotion":emo})

    # Fallback: crawl folders if CSVs absent or yielded nothing
    if not rows:
        for split in ["train","test"]:
            split_dir = base/split
            if not split_dir.exists(): 
                continue
            for d in split_dir.iterdir():
                if not d.is_dir(): 
                    continue
                # accept 1..7 or textual names
                name = d.name.lower()
                digit = None
                m = re.search(r"(?<!\d)([1-7])(?!\d)", name)
                if m:
                    digit = int(m.group(1))
                if digit:
                    emo = {1:"surprise",2:"fear",3:"disgust",4:"happy",5:"sad",6:"angry",7:"neutral"}[digit]
                else:
                    candidates = ["angry","disgust","fear","happy","sad","surprise","neutral",
                                  "anger","happiness","sadness","surprised","disgusted"]
                    emo = next((e for e in candidates if e in name), None)
                    emo = {"anger":"angry","happiness":"happy","sadness":"sad","surprised":"surprise","disgusted":"disgust"}.get(emo, emo)
                if emo not in EMO7: 
                    continue
                for img in d.rglob("*.jpg"):
                    rows.append({"image_path":str(img.resolve()), "emotion":emo})
    return pd.DataFrame(rows)

def main():
    print("Loading UTKFace...")
    df_utk = load_utkface()
    print("UTKFace:", len(df_utk))

    print("Loading FairFace...")
    df_fair = load_fairface()
    print("FairFace:", len(df_fair))

    age_gender = pd.concat([df_utk, df_fair], ignore_index=True).drop_duplicates("image_path")
    out_ag = PROC/"age_gender.csv"
    age_gender.to_csv(out_ag, index=False)
    print("Wrote", out_ag, "rows:", len(age_gender))

    print("Loading RAF-DB...")
    df_raf = load_rafdb()
    out_emo = PROC/"emotion.csv"
    df_raf.to_csv(out_emo, index=False)
    print("Wrote", out_emo, "rows:", len(df_raf))

if __name__ == "__main__":
    main()
