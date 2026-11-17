import io
import numpy as np
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.vector_ar.var_model import VAR

NAMA_BULAN = [
    "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember"
]


def hitung_tahun_bulan(periode: int):
    total_bulan = (2024 * 12 + 2) + (periode - 1)
    tahun = total_bulan // 12
    bulan = NAMA_BULAN[total_bulan % 12]
    return tahun, bulan


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
            raise ValueError("Data tidak cukup atau konstan")
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
