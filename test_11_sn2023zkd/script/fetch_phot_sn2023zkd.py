# fetch_phot_sn2023zkd.py  (REV)
# Baixa fotometria ZTF (IRSA Lightcurve Service) para SN 2023zkd
# Saída: phot.csv  (mjd, band, mag, mag_err)

import csv, io, sys, math, urllib.parse, urllib.request

RA_DEG   = 237.198067   # 15:48:47.536
DEC_DEG  =   9.200078   # +09:12:00.28
R_ARCSEC =   12.0        # raio de busca; tente 5–10″ se 3″ falhar
OUT_CSV  = "phot.csv"

def map_filtercode(code: str) -> str:
    code = (code or "").strip().lower()
    if code.endswith("g"): return "g"
    if code.endswith("r"): return "r"
    if code.endswith("i"): return "i"
    return code[-1:] if code else ""

def http_get(url, timeout=60):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="ignore")

def fetch_lightcurves_irsa(ra_deg, dec_deg, r_arcsec):
    """
    Tenta o endpoint oficial (plural): .../products/lightcurves?POS=ra,dec&SIZE=deg&FORMAT=csv
    Faz fallback para a variante antiga (singular) com R.
    """
    # 1) endpoint oficial (plural): SIZE em graus
    base1 = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/lightcurves"
    params1 = {
        "POS": f"{ra_deg:.6f},{dec_deg:.6f}",
        "SIZE": f"{r_arcsec/3600.0:.8f}",   # graus
        "FORMAT": "csv",
    }
    url1 = base1 + "?" + urllib.parse.urlencode(params1)

    try:
        text = http_get(url1)
        if "mjd" in text.lower() and "filter" in text.lower():
            return text, "lightcurves+SIZE"
    except Exception as e:
        pass

    # 2) fallback (singular): R em graus
    base2 = "https://irsa.ipac.caltech.edu/ibe/search/ztf/products/lightcurve"
    params2 = {
        "POS": f"{ra_deg:.6f},{dec_deg:.6f}",
        "R": f"{r_arcsec/3600.0:.8f}",
        "FORMAT": "csv",
    }
    url2 = base2 + "?" + urllib.parse.urlencode(params2)

    text = http_get(url2)  # pode lançar exceção — deixo propagar para log claro
    return text, "lightcurve+R"

def parse_and_filter(csv_text: str):
    reader = csv.DictReader(io.StringIO(csv_text))
    out, n_all, n_keep = [], 0, 0
    for row in reader:
        n_all += 1
        try:
            # colunas mais comuns no serviço
            mjd  = float(row.get("mjd") or row.get("obsjd") or "")
            mag  = float(row.get("mag") or row.get("magpsf") or "")
            merr = float(row.get("magerr") or row.get("sigmapsf") or "")
            fcod = (row.get("filtercode") or row.get("filter") or row.get("fid") or "")
            band = map_filtercode(fcod)
            # qualidade: catflags==0 (quando presente)
            catflags = row.get("catflags")
            if catflags not in (None, "") and str(catflags).strip() != "0":
                continue
            if not (math.isfinite(mjd) and math.isfinite(mag) and math.isfinite(merr)):
                continue
            if band not in ("g","r","i"):  # só bandas ZTF públicas
                continue
            if merr <= 0 or merr > 1.0:
                continue
            out.append((mjd, band, mag, merr)); n_keep += 1
        except Exception:
            continue
    out.sort(key=lambda x: x[0])
    return out, n_all, n_keep

def write_phot_csv(rows, path=OUT_CSV):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["mjd","band","mag","mag_err"])
        for mjd, band, mag, merr in rows:
            w.writerow([f"{mjd:.5f}", band, f"{mag:.3f}", f"{merr:.3f}"])

def main():
    print(f"[INFO] IRSA query @ RA={RA_DEG:.6f}, Dec={DEC_DEG:.6f}, R={R_ARCSEC}\" …")
    try:
        csv_text, used = fetch_lightcurves_irsa(RA_DEG, DEC_DEG, R_ARCSEC)
    except Exception as e:
        print("[ERROR] Download falhou:", e)
        print("       Tente aumentar R_ARCSEC (p.ex. 10.0) ou verifique sua rede/IRSA.")
        sys.exit(1)

    if not csv_text or "mjd" not in csv_text.lower():
        print("[ERROR] Resposta sem colunas esperadas (mjd,…).")
        sys.exit(1)

    rows, n_all, n_keep = parse_and_filter(csv_text)
    if n_keep == 0:
        # salva raw para inspeção
        with open("ztf_lightcurve_raw.csv", "w", encoding="utf-8") as f:
            f.write(csv_text)
        print("[ERROR] Nenhum ponto útil após filtros. Salvei ztf_lightcurve_raw.csv para checar colunas.")
        print("       Sugestão: aumente R_ARCSEC para 10.0–12.0 e rode de novo.")
        sys.exit(1)

    write_phot_csv(rows, OUT_CSV)
    print(f"[DONE] {OUT_CSV} salvo com {n_keep} pontos (de {n_all} lidos). Endpoint usado: {used}")

if __name__ == "__main__":
    main()