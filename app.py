import io
import numpy as np
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
import streamlit as st

st.set_page_config(page_title="Dashboard Peramalan PO Konveksi", layout="wide")

NAMA_BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]


def hitung_tahun_bulan(periode: int):
    total_bulan = (2024 * 12 + 2) + (periode - 1)
    tahun = total_bulan // 12
    bulan = NAMA_BULAN[total_bulan % 12]
    return tahun, bulan


def inisialisasi_data():
    if "df_bulanan" not in st.session_state:
        po_bulanan = [
            974, 924, 1622, 1301, 736, 710, 1611, 975, 1075, 972,
            1420, 1071, 1137, 1213, 1630, 1411, 877, 1657, 2175, 1539,
            1135, 798, 1207, 1231, 807, 1380, 886, 807, 942, 1052, 2043, 1728
        ]
        rows = []
        for idx, val in enumerate(po_bulanan, start=1):
            thn, bln = hitung_tahun_bulan(idx)
            rows.append({
                "Tahun": thn,
                "Bulan": bln,
                "Periode": idx,
                "Jumlah PO": int(val),
                "Status Data": "Aktual",
            })
        st.session_state.df_bulanan = pd.DataFrame(rows)

    if "df_produk" not in st.session_state:
        data_prod = {
            "WORKJACKET": [
                281, 207, 575, 301, 193, 36, 345, 210, 0, 0,
                579, 313, 237, 862, 217, 455, 228, 259, 391, 511,
                547, 339, 687, 931, 265, 519, 329, 582, 720, 393, 1503, 1396
            ],
            "VARSITY": [
                251, 544, 422, 818, 310, 560, 1203, 441, 594, 300,
                523, 658, 790, 351, 562, 597, 509, 312, 432, 254,
                491, 303, 333, 160, 100, 104, 204, 75, 70, 378, 100, 87
            ],
            "COACH": [
                330, 90, 573, 134, 233, 101, 26, 151, 0, 122,
                37, 64, 110, 0, 0, 0, 0, 0, 89, 197,
                97, 71, 73, 24, 0, 15, 0, 0, 22, 150, 82, 137
            ],
        }
        rows = []
        for idx in range(1, 33):
            thn, bln = hitung_tahun_bulan(idx)
            rows.append({
                "Tahun": thn,
                "Bulan": bln,
                "Periode": idx,
                "WORKJACKET": data_prod["WORKJACKET"][idx - 1],
                "VARSITY": data_prod["VARSITY"][idx - 1],
                "COACH": data_prod["COACH"][idx - 1],
                "Status Data": "Aktual",
            })
        st.session_state.df_produk = pd.DataFrame(rows)
        st.session_state.daftar_produk = ["WORKJACKET", "VARSITY", "COACH"]

    if "mode_edit_bulanan" not in st.session_state:
        st.session_state.mode_edit_bulanan = False
    if "mode_edit_produk" not in st.session_state:
        st.session_state.mode_edit_produk = False


def buat_prediksi_ar_bulanan(df_aktual: pd.DataFrame):
    periode_terakhir = int(df_aktual["Periode"].max()) if not df_aktual.empty else 0
    periode_depan = np.arange(periode_terakhir + 1, periode_terakhir + 4)

    rows_pred = []
    for p in periode_depan:
        thn, bln = hitung_tahun_bulan(int(p))
        rows_pred.append({
            "Tahun": thn,
            "Bulan": bln,
            "Periode": int(p),
            "Status Data": "Prediksi",
        })

    df_pred = pd.DataFrame(rows_pred)
    y = df_aktual["Jumlah PO"].fillna(0).astype(float).values
    lags = min(2, len(y) - 1) if len(y) > 3 else 1

    try:
        if len(y) < 3 or np.std(y) == 0:
            raise ValueError("Data tidak cukup atau konstan untuk AutoReg")
        model = AutoReg(y, lags=lags)
        model_fitted = model.fit()
        pred = model_fitted.predict(start=len(y), end=len(y) + 2)
        df_pred["Jumlah PO"] = np.maximum(0, np.round(pred)).astype(int)

        y_fit = model_fitted.fittedvalues
        mad_val = int(np.round(np.mean(np.abs(y[lags:] - y_fit))))
        model_name = f"AutoReg (Lag {lags})"
    except Exception:
        mean_val = (
            int(np.round(np.mean(y[-3:])))
            if len(y) >= 3
            else (int(np.round(np.mean(y))) if len(y) > 0 else 0)
        )
        df_pred["Jumlah PO"] = mean_val
        mad_val = (
            int(np.round(np.mean(np.abs(y - np.mean(y))))) if len(y) > 0 else 0
        )
        model_name = "AutoReg (Fallback / Rata-rata)"

    df_eval = pd.DataFrame([
        {
            "Metrik / Keterangan": "Rata-rata PO Aktual / Bulan",
            "Jumlah PO": f"{int(np.round(np.mean(y))) if len(y) > 0 else 0:,}",
        },
        {
            "Metrik / Keterangan": "Model yang Digunakan",
            "Jumlah PO": model_name,
        },
        {
            "Metrik / Keterangan": "Rata-rata Meleset / Error (MAD)",
            "Jumlah PO": f"± {mad_val:,} pcs",
        },
    ])
    return (
        df_pred[["Tahun", "Bulan", "Periode", "Jumlah PO", "Status Data"]],
        df_eval,
    )


def buat_prediksi_var_produk(df_aktual: pd.DataFrame, kolom_target: list):
    periode_terakhir = int(df_aktual["Periode"].max()) if not df_aktual.empty else 0
    periode_depan = np.arange(periode_terakhir + 1, periode_terakhir + 4)

    rows_pred = []
    for p in periode_depan:
        thn, bln = hitung_tahun_bulan(int(p))
        rows_pred.append({
            "Tahun": thn,
            "Bulan": bln,
            "Periode": int(p),
            "Status Data": "Prediksi",
        })

    df_pred = pd.DataFrame(rows_pred)
    data_matrix = df_aktual[kolom_target].fillna(0).astype(float)

    row_rata2 = {"Metrik / Keterangan": "Rata-rata PO Aktual / Bulan"}
    row_model_info = {"Metrik / Keterangan": "Model yang Digunakan"}
    row_mad = {"Metrik / Keterangan": "Rata-rata Meleset / Error (MAD)"}

    kolom_aktif = [c for c in kolom_target if data_matrix[c].std() > 0]
    kolom_konstan = [c for c in kolom_target if c not in kolom_aktif]

    var_berhasil = False
    if len(kolom_aktif) >= 2 and len(data_matrix) >= 4:
        try:
            sub_matrix = data_matrix[kolom_aktif]
            model = VAR(sub_matrix)
            lag_order = min(1, len(sub_matrix) - 1)
            results = model.fit(maxlags=lag_order, ic=None)

            lag_value = results.k_ar
            forecast = results.forecast(sub_matrix.values[-lag_value:], steps=3)

            for idx, col in enumerate(kolom_aktif):
                pred_vals = np.maximum(0, np.round(forecast[:, idx])).astype(int)
                df_pred[col] = pred_vals

                y_actual = sub_matrix[col].values
                y_fitted = np.maximum(0, results.fittedvalues[col].values)
                mad_val = int(
                    np.round(np.mean(np.abs(y_actual[lag_value:] - y_fitted)))
                )

                row_rata2[col] = f"{int(np.round(np.mean(y_actual))):,}"
                row_model_info[col] = f"VAR (Lag {lag_value})"
                row_mad[col] = f"± {mad_val:,} pcs"

            var_berhasil = True
        except Exception:
            var_berhasil = False

    kolom_sisa = kolom_konstan if var_berhasil else kolom_target
    for col in kolom_sisa:
        y_col = data_matrix[col].values
        mean_col = (
            int(np.round(np.mean(y_col[-3:])))
            if len(y_col) >= 3
            else (int(np.round(np.mean(y_col))) if len(y_col) > 0 else 0)
        )
        df_pred[col] = mean_col
        row_rata2[col] = f"{int(np.round(np.mean(y_col))) if len(y_col) > 0 else 0:,}"
        row_model_info[col] = "Fallback (Rata-rata)"
        mad_sisa = (
            int(np.round(np.mean(np.abs(y_col - np.mean(y_col)))))
            if len(y_col) > 0
            else 0
        )
        row_mad[col] = f"± {mad_sisa:,} pcs"

    kolom_urut = ["Tahun", "Bulan", "Periode"] + kolom_target + ["Status Data"]
    df_eval = pd.DataFrame([row_rata2, row_model_info, row_mad])
    df_eval = df_eval[["Metrik / Keterangan"] + kolom_target]
    return df_pred[kolom_urut], df_eval


def warnai_baris_prediksi(row):
    if row["Status Data"] == "Prediksi":
        return [
            "background-color: #FEF9C3; color: #1E293B; font-weight: bold"
        ] * len(row)
    return [""] * len(row)


def buat_file_excel(
    df: pd.DataFrame, df_eval: pd.DataFrame, nama_sheet: str, kolom_angka: list
) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=nama_sheet, startrow=0)
        ws = writer.sheets[nama_sheet]

        header_fill = PatternFill(
            start_color="1E3A5F", end_color="1E3A5F", fill_type="solid"
        )
        eval_header_fill = PatternFill(
            start_color="334155", end_color="334155", fill_type="solid"
        )
        header_font = Font(color="FFFFFF", bold=True)
        pred_fill = PatternFill(
            start_color="FEF9C3", end_color="FEF9C3", fill_type="solid"
        )
        pred_font = Font(color="1E293B", bold=True)
        eval_fill = PatternFill(
            start_color="F1F5F9", end_color="F1F5F9", fill_type="solid"
        )
        thin_side = Side(border_style="thin", color="CBD5E1")
        border_style = Border(
            left=thin_side, right=thin_side, top=thin_side, bottom=thin_side
        )

        for col_idx in range(1, len(df.columns) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border_style
        ws.row_dimensions[1].height = 24

        has_status = "Status Data" in df.columns
        idx_status = df.columns.get_loc("Status Data") + 1 if has_status else None

        for row_idx in range(2, len(df) + 2):
            is_prediksi = (
                has_status
                and ws.cell(row=row_idx, column=idx_status).value == "Prediksi"
            )
            for col_idx, col_name in enumerate(df.columns, start=1):
                cell = ws.cell(row=row_idx, column=col_idx)
                cell.border = border_style

                if is_prediksi:
                    cell.fill = pred_fill
                    cell.font = pred_font

                if col_name in kolom_angka:
                    cell.number_format = "#,##0"
                    cell.alignment = Alignment(horizontal="right", vertical="center")
                elif col_name in ["Tahun", "Periode", "Status Data"]:
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                else:
                    cell.alignment = Alignment(horizontal="left", vertical="center")

        start_eval_title = len(df) + 4
        ws.cell(
            row=start_eval_title, column=1, value="2. EVALUASI MODEL STATISTIK"
        ).font = Font(bold=True, size=11, color="1E293B")

        start_eval_table = start_eval_title + 1
        df_eval.to_excel(
            writer, index=False, sheet_name=nama_sheet, startrow=start_eval_table - 1
        )

        for col_idx in range(1, len(df_eval.columns) + 1):
            cell = ws.cell(row=start_eval_table, column=col_idx)
            cell.fill = eval_header_fill
            cell.font = header_font
            cell.alignment = Alignment(
                horizontal="center" if col_idx > 1 else "left", vertical="center"
            )
            cell.border = border_style
        ws.row_dimensions[start_eval_table].height = 22

        for r_offset in range(1, len(df_eval) + 1):
            r_curr = start_eval_table + r_offset
            for col_idx in range(1, len(df_eval.columns) + 1):
                cell = ws.cell(row=r_curr, column=col_idx)
                cell.fill = eval_fill
                cell.border = border_style
                cell.font = Font(bold=(col_idx == 1), color="1E293B")
                cell.alignment = Alignment(
                    horizontal="left" if col_idx == 1 else "right", vertical="center"
                )

        max_cols = max(len(df.columns), len(df_eval.columns))
        for col_idx in range(1, max_cols + 1):
            col_letter = get_column_letter(col_idx)
            max_len = 12
            for row_cells in ws.iter_rows(min_col=col_idx, max_col=col_idx):
                for c in row_cells:
                    if c.value is not None and c.row != start_eval_title:
                        max_len = max(max_len, len(str(c.value)) + 4)
            ws.column_dimensions[col_letter].width = max_len

        ws.freeze_panes = "A2"

    return output.getvalue()


inisialisasi_data()

st.sidebar.title("📌 Menu Navigasi")
halaman = st.sidebar.radio(
    "Pilih Halaman:", ["1. PO Bulanan", "2. Peramalan Produk Utama"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("📂 Import Data dari Excel / CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload file template Excel (.xlsx) atau CSV", type=["xlsx", "csv"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            df_up = pd.read_csv(uploaded_file)
        else:
            df_up = pd.read_excel(uploaded_file)

        df_up.columns = [str(c).strip() for c in df_up.columns]

        if "Status Data" in df_up.columns:
            df_up = df_up[
                df_up["Status Data"].astype(str).str.strip().str.lower() == "aktual"
            ].copy()

        if "Periode" in df_up.columns:
            df_up["Periode"] = pd.to_numeric(df_up["Periode"], errors="coerce")
            df_up = df_up.dropna(subset=["Periode"]).copy()
            df_up["Periode"] = df_up["Periode"].astype(int)

        if "Periode" in df_up.columns:
            thn_bln_auto = [hitung_tahun_bulan(p) for p in df_up["Periode"]]
            if "Tahun" not in df_up.columns:
                df_up["Tahun"] = [tb[0] for tb in thn_bln_auto]
            else:
                df_up["Tahun"] = pd.to_numeric(
                    df_up["Tahun"], errors="coerce"
                ).ffill()
                df_up["Tahun"] = df_up["Tahun"].fillna(
                    pd.Series([tb[0] for tb in thn_bln_auto], index=df_up.index)
                ).astype(int)

            if "Bulan" not in df_up.columns:
                df_up["Bulan"] = [tb[1] for tb in thn_bln_auto]
            else:
                df_up["Bulan"] = df_up["Bulan"].fillna(
                    pd.Series([tb[1] for tb in thn_bln_auto], index=df_up.index)
                )

        mode_import = st.sidebar.radio(
            "Pilih cara memproses file:",
            [
                "Tambahkan data baru ke dashboard",
                "Ganti seluruh data dengan file ini",
            ],
        )

        if st.sidebar.button("🚀 Proses File Excel", use_container_width=True):
            if "Periode" not in df_up.columns:
                st.sidebar.error("File harus memiliki kolom 'Periode'!")
            else:
                berhasil_proses = False

                if "Jumlah PO" in df_up.columns:
                    df_b_baru = df_up.copy()
                    df_b_baru["Jumlah PO"] = pd.to_numeric(
                        df_b_baru["Jumlah PO"], errors="coerce"
                    )
                    df_b_baru = df_b_baru.dropna(subset=["Jumlah PO"]).copy()
                    df_b_baru["Jumlah PO"] = df_b_baru["Jumlah PO"].astype(int)
                    df_b_baru["Status Data"] = "Aktual"
                    df_b_baru = df_b_baru[
                        ["Tahun", "Bulan", "Periode", "Jumlah PO", "Status Data"]
                    ]

                    if mode_import == "Tambahkan data baru ke dashboard":
                        gabung = pd.concat(
                            [st.session_state.df_bulanan, df_b_baru],
                            ignore_index=True,
                        )
                        gabung = (
                            gabung.drop_duplicates(subset=["Periode"], keep="last")
                            .sort_values("Periode")
                            .reset_index(drop=True)
                        )
                        st.session_state.df_bulanan = gabung
                    else:
                        st.session_state.df_bulanan = df_b_baru.sort_values(
                            "Periode"
                        ).reset_index(drop=True)
                    berhasil_proses = True

                kolom_meta = {
                    "Tahun",
                    "Bulan",
                    "Periode",
                    "Jumlah PO",
                    "Status Data",
                }
                kolom_prod_file = [
                    c
                    for c in df_up.columns
                    if c not in kolom_meta and not c.startswith("Unnamed")
                ]

                if kolom_prod_file:
                    df_p_baru = df_up.copy()
                    for c in kolom_prod_file:
                        df_p_baru[c] = pd.to_numeric(
                            df_p_baru[c], errors="coerce"
                        )
                    df_p_baru = df_p_baru.dropna(
                        subset=kolom_prod_file, how="all"
                    ).copy()
                    for c in kolom_prod_file:
                        df_p_baru[c] = df_p_baru[c].fillna(0).astype(int)
                    df_p_baru["Status Data"] = "Aktual"
                    df_p_baru = df_p_baru[
                        ["Tahun", "Bulan", "Periode"]
                        + kolom_prod_file
                        + ["Status Data"]
                    ]

                    if mode_import == "Tambahkan data baru ke dashboard":
                        gabung_p = pd.concat(
                            [st.session_state.df_produk, df_p_baru],
                            ignore_index=True,
                        )
                        gabung_p = (
                            gabung_p.drop_duplicates(subset=["Periode"], keep="last")
                            .sort_values("Periode")
                            .reset_index(drop=True)
                        )
                        for c in kolom_prod_file:
                            if c not in st.session_state.daftar_produk:
                                st.session_state.daftar_produk.append(c)
                            gabung_p[c] = gabung_p[c].fillna(0).astype(int)
                        st.session_state.df_produk = gabung_p[
                            ["Tahun", "Bulan", "Periode"]
                            + st.session_state.daftar_produk
                            + ["Status Data"]
                        ]
                    else:
                        st.session_state.daftar_produk = kolom_prod_file
                        st.session_state.df_produk = df_p_baru.sort_values(
                            "Periode"
                        ).reset_index(drop=True)
                    berhasil_proses = True

                if berhasil_proses:
                    st.sidebar.success(
                        "Data dari file berhasil diproses ke dashboard!"
                    )
                    st.rerun()
                else:
                    st.sidebar.warning(
                        "Tidak ditemukan kolom 'Jumlah PO' atau kolom produk yang valid."
                    )

    except Exception as e:
        st.sidebar.error(f"Gagal membaca file: {e}")

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Reset ke Data Awal (Periode 1–32)"):
    for key in [
        "df_bulanan",
        "df_produk",
        "daftar_produk",
        "mode_edit_bulanan",
        "mode_edit_produk",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    st.rerun()

# ==============================================================================
# HALAMAN 1: PO BULANAN
# ==============================================================================
if halaman == "1. PO Bulanan":
    st.title("📦 Rekap & Peramalan PO Bulanan (Model AR)")
    st.caption(
        "Menampilkan rekap historis dan proyeksi 3 bulan ke depan menggunakan Autoregresi (AR)."
    )

    df_akt = st.session_state.df_bulanan.copy()
    df_pred, df_eval_bulanan = buat_prediksi_ar_bulanan(df_akt)
    df_gabungan = pd.concat([df_akt, df_pred], ignore_index=True)

    baris_pred_1 = df_pred.iloc[0]
    p_target = int(baris_pred_1["Periode"])
    bln_target = baris_pred_1["Bulan"]
    thn_target = int(baris_pred_1["Tahun"])
    val_pred_awal = int(baris_pred_1["Jumlah PO"])

    with st.container(border=True):
        st.subheader(
            f"✏️ Input Realisasi Aktual: {bln_target} {thn_target} (Periode {p_target})"
        )
        st.write(
            f"Angka prediksi AR saat ini untuk **{bln_target} {thn_target}** adalah **{val_pred_awal:,} pcs**. "
            "Masukkan angka realisasi jika sudah tersedia:"
        )
        col_in, col_btn = st.columns([2, 1])
        with col_in:
            nilai_aktual_baru = st.number_input(
                f"Jumlah PO Aktual ({bln_target} {thn_target}):",
                min_value=0,
                value=val_pred_awal,
                step=10,
                key=f"input_
