# -*- coding: utf-8 -*-
"""
Gera tabelas (CSV/MD) e também PNGs a partir dos resultados do ajuste multivariado.
Uso:
    python make_tables.py <OUTDIR>
Exemplo:
    python make_tables.py results_los6
Arquivos esperados em <OUTDIR> (os que existirem serão usados):
    - summary.json                       # métricas globais (R2, deltaR2, p permut.)
    - ols_std_coefs.csv                  # betas padronizados (OLS) com colunas 'beta_<col>'
    - ols_std_pvalues.json               # p-values (se existir)
    - per_event_summary.csv              # resumo por evento (n, SNR_p50, LOS_p50, ...)
Saídas geradas em <OUTDIR>:
    - table_coefficients.(csv|md|png)
    - table_metrics.(csv|md|png)
    - table_events.(csv|md|png)          # se per_event_summary.csv existir
"""

import sys, os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def df_to_png(df, outpath, title=None, fontsize=10):
    """Renderiza um DataFrame como PNG usando matplotlib."""
    # tamanho automático: largura ~ número de colunas, altura ~ número de linhas
    fig_w = max(6.0, len(df.columns) * 1.8)
    fig_h = max(2.5, len(df) * 0.55 + 1.3)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis('off')
    tbl = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center'
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.scale(1.2, 1.2)
    if title:
        plt.title(title, fontsize=fontsize+2, pad=10)
    plt.savefig(outpath, dpi=220, bbox_inches="tight")
    plt.close(fig)

def safe_load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def main():
    if len(sys.argv) < 2:
        print("Uso: python make_tables.py <OUTDIR>")
        sys.exit(1)

    OUTDIR = sys.argv[1]
    if not os.path.isdir(OUTDIR):
        print(f"[ERRO] Pasta não encontrada: {OUTDIR}")
        sys.exit(1)

    # ----- caminhos de entrada
    f_summary   = os.path.join(OUTDIR, "summary.json")
    f_coefs     = os.path.join(OUTDIR, "ols_std_coefs.csv")
    f_pvals     = os.path.join(OUTDIR, "ols_std_pvalues.json")
    f_events    = os.path.join(OUTDIR, "per_event_summary.csv")

    # ----- COEFICIENTES (OLS padronizado)
    coef_df = pd.DataFrame()
    if os.path.exists(f_coefs):
        raw = pd.read_csv(f_coefs)
        # Espera uma única linha com colunas beta_<nome>
        row = raw.iloc[0].to_dict()
        # normaliza nomes: beta_log_DL -> log_DL
        renamed = {k.replace("beta_", ""): v for k, v in row.items() if k.startswith("beta_")}
        coef_df = pd.DataFrame([renamed]).T.reset_index()
        coef_df.columns = ["predictor", "standardized_beta"]
        # adiciona p-values se houver
        pvals = safe_load_json(f_pvals)
        if pvals:
            coef_df["p_value"] = coef_df["predictor"].map(pvals).replace({None: np.nan})
        # ordena por magnitude
        coef_df["abs_beta"] = coef_df["standardized_beta"].abs()
        coef_df = coef_df.sort_values("abs_beta", ascending=False).drop(columns=["abs_beta"])
    else:
        print("[AVISO] ols_std_coefs.csv não encontrado — pulando tabela de coeficientes.")

    # ----- MÉTRICAS GLOBAIS
    metrics = pd.DataFrame()
    summ = safe_load_json(f_summary)
    if summ:
        # nomes amigáveis
        nice = {
            "R2_adj_std": "Adj. R² (padronizado)",
            "delta_R2_LOS": "ΔR² (adicionar LOS)",
            "R2_full": "R² (modelo completo)",
            "perm_deltaR2_obs": "ΔR² observado (permutação LOS)",
            "perm_pvalue": "p-valor permutação (LOS)"
        }
        items = []
        for k, v in summ.items():
            name = nice.get(k, k)
            # formatação numérica
            if isinstance(v, (int, float)):
                if abs(v) >= 1000:
                    sval = f"{v:,.0f}"
                else:
                    sval = f"{v:.4f}"
            else:
                sval = str(v)
            items.append({"metric": name, "value": sval})
        metrics = pd.DataFrame(items)
    else:
        print("[AVISO] summary.json não encontrado — pulando tabela de métricas.")

    # ----- RESUMO POR EVENTO
    events_df = pd.DataFrame()
    if os.path.exists(f_events):
        events_df = pd.read_csv(f_events)
        # ordem mais informativa: por n desc e depois SNR_p50
        sort_cols = [c for c in ["n", "SNR_p50"] if c in events_df.columns]
        if sort_cols:
            events_df = events_df.sort_values(sort_cols, ascending=[False] * len(sort_cols))
        # formatação leve
        for col in events_df.columns:
            if events_df[col].dtype.kind in "if":
                events_df[col] = np.round(events_df[col].astype(float), 4)
    else:
        print("[AVISO] per_event_summary.csv não encontrado — pulando tabela de eventos.")

    # ----- SALVAR CSV / MD / PNG
    # 1) Coeficientes
    if not coef_df.empty:
        coef_csv = os.path.join(OUTDIR, "table_coefficients.csv")
        coef_md  = os.path.join(OUTDIR, "table_coefficients.md")
        coef_png = os.path.join(OUTDIR, "table_coefficients.png")
        coef_df.to_csv(coef_csv, index=False)
        coef_df.to_markdown(coef_md, index=False)
        df_to_png(coef_df, coef_png, title="Standardized Coefficients", fontsize=10)
        print(f"[OK] coeficientes -> {coef_csv}, {coef_md}, {coef_png}")

    # 2) Métricas
    if not metrics.empty:
        met_csv = os.path.join(OUTDIR, "table_metrics.csv")
        met_md  = os.path.join(OUTDIR, "table_metrics.md")
        met_png = os.path.join(OUTDIR, "table_metrics.png")
        metrics.to_csv(met_csv, index=False)
        metrics.to_markdown(met_md, index=False)
        df_to_png(metrics, met_png, title="Model Metrics", fontsize=10)
        print(f"[OK] métricas -> {met_csv}, {met_md}, {met_png}")

    # 3) Eventos
    if not events_df.empty:
        ev_csv = os.path.join(OUTDIR, "table_events.csv")
        ev_md  = os.path.join(OUTDIR, "table_events.md")
        ev_png = os.path.join(OUTDIR, "table_events.png")
        events_df.to_csv(ev_csv, index=False)
        events_df.to_markdown(ev_md, index=False)
        df_to_png(events_df, ev_png, title="Per-event summary", fontsize=9)
        print(f"[OK] eventos -> {ev_csv}, {ev_md}, {ev_png}")

    print("[DONE] Tabelas geradas.")

if __name__ == "__main__":
    main()