# -*- coding: utf-8 -*-
import os, argparse
import numpy as np
import pandas as pd

C_KMS = 299792.458

def read_csv_req(path, apelido):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[erro] não encontrei {apelido}: {path}")
    return pd.read_csv(path)

def find_col_name(df, candidates, required=True):
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Coluna não encontrada; tentei: {candidates}")
    return None

def main():
    ap = argparse.ArgumentParser(description="CET LOS density usando chi=(c/H0)*z")
    ap.add_argument("--pantheon", required=True)
    ap.add_argument("--pairs",    required=True)
    ap.add_argument("--outdir",   required=True)
    ap.add_argument("--H0", type=float, default=70.0)
    ap.add_argument("--min_Rap", type=float, default=0.5)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # Ler
    sn = read_csv_req(args.pantheon, "Pantheon")
    pr = read_csv_req(args.pairs, "pares 2MRS")

    # Mapear nomes (apenas nomes!)
    sn_id_name = find_col_name(sn, ["CID","ID","sn_id","SN_ID"])
    sn_z_name  = find_col_name(sn, ["zCMB","z","zHD","Z"])

    pr_sn_id_name = find_col_name(pr, ["SN_ID","sn_id","CID","ID"])
    pr_sn_z_name  = find_col_name(pr, ["SN_z","sn_z","zSN","z_sn"], required=False)
    g_z_name      = find_col_name(pr, ["G_z","z_g","Z_NEI","nei_z","Z_gal"])
    rproj_name    = find_col_name(pr, ["Rproj_Mpc","RPROJ_MPC","R_proj_Mpc","Rproj"])
    kmag_name     = find_col_name(pr, ["G_K","Kmag","Kcmag","Kmag_2MRS","K"], required=False)

    print(f"[cols] SN_ID:{pr_sn_id_name} | G_z:{g_z_name} | Rproj:{rproj_name} | SN_z:{pr_sn_z_name or '(merge)'} | G_K:{kmag_name or 'N/A'}")

    # Adicionar SN_z aos pares se necessário
    if pr_sn_z_name is None:
        pr = pr.merge(sn[[sn_id_name, sn_z_name]],
                      left_on=pr_sn_id_name, right_on=sn_id_name,
                      how="left")
        pr.rename(columns={sn_z_name:"SN_z"}, inplace=True)
        pr_sn_z_name = "SN_z"

    # Distância com métrica CET (chi = (c/H0)*z) por SN
    sn_map = pd.DataFrame({
        "SN_ID": sn[sn_id_name],
        "zSN":   sn[sn_z_name].astype(float),
        "chi_Mpc": (C_KMS/args.H0) * sn[sn_z_name].astype(float)
    })

    # Re-leituras SEMPRE a partir de pr (agora já definitivo)
    g_z   = pr[g_z_name].astype(float).values
    sn_zp = pr[pr_sn_z_name].astype(float).values
    rproj = pr[rproj_name].astype(float).values
    if kmag_name:
        k_mag = pr[kmag_name].astype(float).values
    else:
        k_mag = None

    # Manter galáxias entre nós e a SN
    mask = (g_z > 0) & (sn_zp > 0) & (g_z < sn_zp)
    pr_keep = pr.loc[mask].copy()

    # Raio projetado por SN (máx observado; piso min_Rap)
    pr_keep["Rproj_Mpc"] = rproj[mask]
    rap_by_sn = pr_keep.groupby(pr_sn_id_name)["Rproj_Mpc"].max().clip(lower=args.min_Rap)

    # Peso fotométrico K (opcional)
    if k_mag is not None:
        pr_keep["wK"] = 10.0 ** (-0.4 * k_mag[mask])
    else:
        pr_keep["wK"] = 1.0

    # Agregar por SN
    counts = pr_keep.groupby(pr_sn_id_name).agg(
        N=("Rproj_Mpc","size"),
        sum_wK=("wK","sum")
    ).reset_index().rename(columns={pr_sn_id_name:"SN_ID"})

    # Juntar chi e raio
    out = counts.merge(sn_map, on="SN_ID", how="left")
    out["R_ap_Mpc"] = out["SN_ID"].map(rap_by_sn).fillna(args.min_Rap)

    # Vol e densidades
    out["Vol_LOS_Mpc3"] = np.pi * (out["R_ap_Mpc"]**2) * out["chi_Mpc"]
    out["dens_LOS"]     = out["N"]      / out["Vol_LOS_Mpc3"]
    out["dens_LOS_Kw"]  = out["sum_wK"] / out["Vol_LOS_Mpc3"]

    # Salvar
    pairs_out = os.path.join(args.outdir, "los_pairs_kept.csv")
    out_sn    = os.path.join(args.outdir, "los_by_sn.csv")
    pr_keep.to_csv(pairs_out, index=False)
    out.to_csv(out_sn, index=False)

    n_sn_tot = sn.shape[0]
    n_sn_los = out["SN_ID"].nunique()
    print(f"[LOS CET] SNs total: {n_sn_tot} | com LOS>0: {n_sn_los} ({100.0*n_sn_los/n_sn_tot:.1f}%)")
    print(f"[salvo] {out_sn}")
    print(f"[salvo] {pairs_out}")

if __name__ == "__main__":
    main()