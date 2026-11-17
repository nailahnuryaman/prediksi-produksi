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
            is_prediksi = ws.cell(row=row_idx, column=idx_status).value == "Prediksi"
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
        ws.cell(row=start_eval_title, column=1, value="2. EVALUASI MODEL STATISTIK").font = Font(bold=True, size=11, color="1E293B")

        start_eval_table = start_eval_title + 1
        df_eval.to_excel(writer, index=False, sheet_name=nama_sheet, startrow=start_eval_table - 1)

        for col_idx in range(1, len(df_eval.columns) + 1):
            cell = ws.cell(row=start_eval_table, column=col_idx)
            cell.fill = eval_header_fill
            cell.font = header_font
            cell.alignment = Alignment(horizontal="center" if col_idx > 1 else "left", vertical="center")
            cell.border = border_style
        ws.row_dimensions[start_eval_table].height = 22

        for r_offset in range(1, len(df_eval) + 1):
            r_curr = start_eval_table + r_offset
            for col_idx in range(1, len(df_eval.columns) + 1):
                cell = ws.cell(row=r_curr, column=col_idx)
                cell.fill = eval_fill
                cell.border = border_style
                cell.font = Font(bold=(col_idx == 1), color="1E293B")
                cell.alignment = Alignment(horizontal="left" if col_idx == 1 else "right", vertical="center")

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
