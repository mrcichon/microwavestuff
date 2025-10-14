import tempfile
import pandas as pd
import numpy as np
import skrf as rf
from pathlib import Path
import re
import io

def loadFile(p):
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    l = None
    
    for encoding in encodings:
        try:
            with open(p, "r", encoding=encoding) as f:
                l = f.readlines()
                break
        except UnicodeDecodeError:
            continue
    
    if l is None:
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            l = f.readlines()
    
    fx = []
    for ln in l:
        fx.append(ln if ln.lstrip().startswith(("!", "#")) else ln.replace(",", "."))
    
    t = tempfile.NamedTemporaryFile("w+", delete=False, suffix=Path(p).suffix, encoding="utf-8")
    t.writelines(fx)
    t.flush()
    t.close()
    
    network = rf.Network(t.name)
    return network

def parse_polar_rms(path):
    """Parse MegiQ RMS files - handles both linear (HV) and circular (RHCP/LHCP) polarization"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    
    raw = raw.lstrip("\ufeff")
    text = raw.replace("\r\n", "\n").replace(",", ".")
    lines = text.split("\n")
    
    header_idx = None
    is_circular = False
    
    for i, line in enumerate(lines):
        if "RHCP" in line and "LHCP" in line:
            if "dB.YZ" in line or "dB.ZX" in line or "dB.XY" in line:
                header_idx = i
                is_circular = True
                break
    
    if header_idx is None:
        for i, line in enumerate(lines):
            if "dB.YZ.H" in line and "dB.ZX.H" in line and "dB.XY.H" in line:
                header_idx = i
                is_circular = False
                break
    
    if header_idx is None:
        raise ValueError("No valid RMS header found (neither HV nor RHCP/LHCP)")

    header_line = lines[header_idx]
    hdr_tokens = [t for t in header_line.split("\t")]
    colnames = ["angle_deg"] + [t for t in hdr_tokens[1:] if t]
    
    data_block = "\n".join(lines[header_idx + 1:])
    df = pd.read_csv(io.StringIO(data_block), sep=r"\s+|\t+", engine="python", names=colnames)
    
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["angle_deg"]).reset_index(drop=True)
    
    if is_circular:
        def calc_rl_sum(rhcp_series, lhcp_series):
            if rhcp_series is None or lhcp_series is None:
                return None
            rhcp_linear = 10**(rhcp_series / 10.0)
            lhcp_linear = 10**(lhcp_series / 10.0)
            rl_power = rhcp_linear + lhcp_linear
            return 10 * np.log10(rl_power)
        
        return {
            "angle_deg": df["angle_deg"],
            "polarization": "circular",
            "YZ": {
                "RHCP": df.get("dB.YZ.RHCP"),
                "LHCP": df.get("dB.YZ.LHCP"),
                "RL": calc_rl_sum(df.get("dB.YZ.RHCP"), df.get("dB.YZ.LHCP"))
            },
            "ZX": {
                "RHCP": df.get("dB.ZX.RHCP"),
                "LHCP": df.get("dB.ZX.LHCP"),
                "RL": calc_rl_sum(df.get("dB.ZX.RHCP"), df.get("dB.ZX.LHCP"))
            },
            "XY": {
                "RHCP": df.get("dB.XY.RHCP"),
                "LHCP": df.get("dB.XY.LHCP"),
                "RL": calc_rl_sum(df.get("dB.XY.RHCP"), df.get("dB.XY.LHCP"))
            },
        }
    else:
        return {
            "angle_deg": df["angle_deg"],
            "polarization": "linear",
            "YZ": {"HV": df.get("dB.YZ.HV")},
            "ZX": {"HV": df.get("dB.ZX.HV")},
            "XY": {"HV": df.get("dB.XY.HV")},
        }

def parse_polar_pustelnik(path):
    """Parse Pustelnik format (angle + HV values)"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    raw = raw.lstrip("\ufeff").replace("\r\n", "\n").replace(",", ".")
    lines = raw.split("\n")
    
    header_idx = None
    for i, line in enumerate(lines):
        if re.search(r"HV", line, re.I) and ("dB." in line or "dB" in line):
            header_idx = i
            break
    
    if header_idx is not None:
        header_line = lines[header_idx]
        hdr_tokens = [t for t in header_line.split("\t")]
        colnames = ["angle_deg"] + [t for t in hdr_tokens[1:] if t]
        data_block = "\n".join(lines[header_idx + 1:])
        df = pd.read_csv(io.StringIO(data_block), sep=r"\s+|\t+", engine="python", names=colnames)
        
        hv_col = None
        for c in df.columns:
            if re.search(r"HV$", c, re.I):
                hv_col = c
                break
        if hv_col is None and "HV" in df.columns:
            hv_col = "HV"
        if hv_col is None and len(df.columns) >= 2:
            hv_col = df.columns[1]
        
        df = df.dropna(subset=["angle_deg"]).reset_index(drop=True)
        return {
            "angle_deg": pd.to_numeric(df["angle_deg"], errors="coerce"),
            "HV": pd.to_numeric(df[hv_col], errors="coerce")
        }
    else:
        start = 0
        number_line = re.compile(r"^\s*-?\d+(\.\d+)?(\s+|-?\d|\.)+")
        for i, ln in enumerate(lines):
            if number_line.match(ln.strip()):
                start = i
                break
        block = "\n".join(l for l in lines[start:] if l.strip())
        df = pd.read_csv(io.StringIO(block), sep=r"\s+|\t+", engine="python", header=None)
        
        if df.shape[1] < 2:
            raise ValueError("Pustelnik file must have at least 2 columns: angle and HV")
        
        df = df.iloc[:, :2]
        df.columns = ["angle_deg", "HV"]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["angle_deg", "HV"]).reset_index(drop=True)
        return {"angle_deg": df["angle_deg"], "HV": df["HV"]}


def parse_theta_phi_file(path):
    """Parse Theta/Phi files with format: Theta [deg], Phi [deg], Abs(Dir.)[dBi]"""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()
    
    raw = raw.lstrip("\ufeff").replace("\r\n", "\n").replace(",", ".")
    lines = [ln for ln in raw.split("\n") if ln.strip() and not set(ln.strip()) <= set("- ")]
    
    header_idx = None
    for i, ln in enumerate(lines):
        if "Theta" in ln and "Phi" in ln:
            header_idx = i
            break
    
    data_lines = lines[header_idx + 1:] if header_idx is not None else lines
    
    float_re = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
    rows = []
    for ln in data_lines:
        nums = [float(x) for x in float_re.findall(ln)]
        if len(nums) >= 3:
            rows.append((nums[0], nums[1], nums[2]))
    
    if not rows:
        raise ValueError("No numeric data found in theta/phi file")
    
    df = pd.DataFrame(rows, columns=["theta_deg", "phi_deg", "dir_dbi"])
    return df
