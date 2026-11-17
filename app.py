import io
import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR
import plotly.graph_objects as go
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

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
            1420, 1071, 1137, 1213, 1630, 1411, 877, 1657, 2175, 1539, 1135, 798,
            1207, 1231, 807, 1380, 886, 807, 942, 1052, 2043, 1728
        ]
        rows = []
        for idx, val in enumerate(po_bulanan, start=1):
            thn, bln = hitung_tahun_bulan(idx)
            rows.append({
                "Tahun": thn,
                "Bulan": bln,
                "Periode": idx,
                "Jumlah PO": int(val),
                "Status Data": "Aktual"
            })
        st.session_state.df_bulanan = pd.DataFrame(rows)

    if "df_produk" not in st.session_state:
        data_prod = {
            "WORKJACKET": [
                281, 207, 575, 301, 193, 36, 345, 210, 0, 0,
                579, 313, 237, 862, 217, 455, 228, 259, 391, 511, 547, 339,
                687, 931, 265, 519, 329, 582, 720, 393, 1503, 1396
            ],
            "VARSITY": [
                251, 544, 422, 818, 310, 560, 1203, 441, 594, 300,
                523, 658, 790, 351, 562, 597, 509, 312, 432, 254, 491, 303,
                333, 160, 100, 104, 204, 75, 70, 378, 100, 87
            ],
            "COACH": [
                330, 90, 573, 134, 233, 101, 26, 151, 0, 122,
                37, 64, 110, 0, 0, 0, 0, 0, 89, 197, 97, 71,
                73, 24, 0, 15, 0, 0, 22, 150, 82, 137
            ]
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
                "Status Data": "Aktual"
            })
        st.session_state.df_produk = pd.DataFrame(rows)
        st.session_state.daftar_produk = ["WORKJACKET", "VARSITY", "COACH"]

    if "mode_edit_bulanan" not in st.session_state:
        st.session_state.mode_edit_bulanan = False
    if "mode_edit_produk" not in st.session_state:
        st.session_state.mode_edit_produk = False

def buat_prediksi_ar_bulanan(df_aktual: pd.DataFrame):
    periode_terakhir = int(df_aktual["Periode"].max())
    periode_depan = np.arange(periode_terakhir + 1, periode_terakhir + 4)

    rows_pred = []
    for p in periode_depan:
        thn, bln = hitung_tahun_bulan(int(p))
        rows_pred.append({
            "Tahun": thn,
            "Bulan": bln,
            "Periode": int(p),
            "Status Data": "Prediksi"
        })

    df_pred = pd.DataFrame(rows_pred)
    y = df_aktual["Jumlah PO"].fillna(0).astype(float).values
    lags = min(2, len(y) - 1) if len(y) > 3 else 1

    try:
        model = AutoReg(y, lags=lags)
        model_fitted = model.fit()
        pred = model_fitted.predict(start=len(y), end=len(y) + 2)
        df_pred["Jumlah PO"] = np.maximum(0, np.round(pred)).astype(int)
        
        y_fit = model_fitted.fittedvalues
        mad_val = int(np.round(np.mean(np.abs(y[lags:] - y_fit))))
        model_name = f"AutoReg (Lag {lags})"
    except Exception:
        mean_val = int(np.round(np.mean(y[-3:]))) if len(y) >= 3 else int(np.round(np.mean(y)))
        df_pred["Jumlah PO"] = mean_val
        mad_val = int(np.round(np.mean(np.abs(y - np.mean(y)))))
        model_name = "AutoReg (Fallback)"

    df_eval = pd.DataFrame([
        {"Metrik / Keterangan": "Rata-rata PO Aktual / Bulan", "Jumlah PO": f"{int(np.round(np.mean(y))):,}"},
        {"Metrik / Keterangan": "Model yang Digunakan", "Jumlah PO": model_name},
        {"Metrik / Keterangan": "Rata-rata Meleset / Error (MAD)", "Jumlah PO": f"± {mad_val:,} pcs"}
    ])
    return df_pred[["Tahun", "Bulan", "Periode", "Jumlah PO", "Status Data"]], df_eval

def buat_prediksi_var_produk(df_aktual: pd.DataFrame, kolom_target: list):
    periode_terakhir = int(df_aktual["Periode"].max())
    periode_depan = np.arange(periode_terakhir + 1, periode_terakhir + 4)

    rows_pred = []
    for p in periode_depan:
        thn, bln = hitung_tahun_bulan(int(p))
        rows_pred.append({
            "Tahun": thn,
            "Bulan": bln,
            "Periode": int(p),
            "Status Data": "Prediksi"
        })

    df_pred = pd.DataFrame(rows_pred)
    data_matrix = df_aktual[kolom_target].fillna(0).astype(float)

    row_rata2 = {"Metrik / Keterangan": "Rata-rata PO Aktual / Bulan"}
    row_model_info = {"Metrik / Keterangan": "Model yang Digunakan"}
    row_mad = {"Metrik / Keterangan": "Rata-rata Meleset / Error (MAD)"}

    try:
        model = VAR(data_matrix)
        lag_order = min(1, len(data_matrix) - 1)
        results = model.fit(maxlags=lag_order, ic=None)
        
        lag_value = results.k_ar
        forecast = results.forecast(data_matrix.values[-lag_value:], steps=3)
        
        for idx, col in enumerate(kolom_target):
            pred_vals = np.maximum(0, np.round(forecast[:, idx])).astype(int)
            df_pred[col] = pred_vals
            
            y_actual = data_matrix[col].values
            y_fitted = np.maximum(0, results.fittedvalues[col].values)
            mad_val = int(np.round(np.mean(np.abs(y_actual[lag_value:] - y_fitted))))
            
            row_rata2[col] = f"{int(np.round(np.mean(y_actual))):,}"
            row_model_info[col] = f"VAR (Lag {lag_value})"
            row_mad[col] = f"± {mad_val:,} pcs"
            
    except Exception:
        for col in kolom_target:
            y_col = data_matrix[col].values
            mean_col = int(np.round(np.mean(y_col[-3:]))) if len(y_col) >= 3 else int(np.round(np.mean(y_col)))
            df_pred[col] = mean_col
            row_rata2[col] = f"{int(np.round(np.mean(y_col))):,}"
            row_model_info[col] = "VAR (Fallback)"
            row_mad[col] = f"± {int(np.round(np.mean(np.abs(y_col - np.mean(y_col))))):,} pcs"

    kolom_urut = ["Tahun", "Bulan", "Periode"] + kolom_target + ["Status Data"]
    df_eval = pd.DataFrame([row_rata2, row_model_info, row_mad])
    return df_pred[kolom_urut], df_eval

def warnai_baris_prediksi(row):
    if row["Status Data"] == "Prediksi":
        return ["background-color: #FEF9C3; color: #1E293B; font-weight: bold"] * len(row)
    return [""] * len(row)

def buat_file_excel(df: pd.DataFrame, df_eval: pd.DataFrame, nama_sheet: str, kolom_angka: list) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=nama_sheet, startrow=0)
        ws = writer.sheets[nama_sheet]

        header_fill = PatternFill(start_color="1E3A5F", end_color="1E3A5F", fill_type="solid")
        eval_header_fill = PatternFill(start_color="334155", end_color="334155", fill_type="solid")
        header_font = Font(color="FFFFFF", bold=True)
        pred_fill = PatternFill(start_color="FEF9C3", end_color="FEF9C3", fill_type="solid")
        pred_font = Font(color="1E293B", bold=True)
        eval_fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
        thin_side = Side(border_style="thin", color="CBD5E1")
        border_style = Border(left=thin_side, right=thin_side, top=thin_side, bottom=thin_side)

        for col_idx in range(1, len(df.columns) + 1):
            cell = ws.cell(row=1, column=col_idx)
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = border_style
        ws.row_dimensions[1].height = 24

        idx_status = df.columns.get_loc("Status Data") + 1
        for row_idx in range(2, len(df) + 2):
            is_prediksi = ws.cell(row=row
