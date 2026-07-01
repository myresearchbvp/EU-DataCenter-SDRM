# -*- coding: utf-8 -*-
"""01_build_eurostat_decision_matrix.ipynb

built in Colab.

"""

# @title
# -*- coding: utf-8 -*-
"""00_build_eurostat_decision_matrix_flexible.ipynb"""

# ====================================================================
# INSTALL DEPENDENCIES
# ====================================================================
import subprocess
import sys

def install_deps():
    try:
        import eurostat
    except ImportError:
        print("Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "eurostat", "pandas", "openpyxl"])

install_deps()

# ====================================================================
# IMPORTS & SETUP
# ====================================================================
import pandas as pd
import numpy as np
import eurostat
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
import io
import base64
import re
from datetime import datetime

# ====================================================================
# 1. FIXED DEFINITIONS & CONSTANTS
# ====================================================================
LOCKED_24_REGIONS = [
    'IE06', 'DE71', 'DE30', 'NL32', 'FR10', 'SE11', 'FI1B', 'DK01', 'ES30',
    'ITC4', 'AT13', 'BE10', 'CZ01', 'PL91', 'RO32', 'RO11', 'HU11', 'BG41',
    'EL30', 'PT17', 'EE00', 'LT01', 'SK01', 'LV00'
]

REQUIRED_NUTS0 = list(set([r[:2] if not r.startswith('EL') else 'EL' for r in LOCKED_24_REGIONS]))

CRITERIA_META = {
    'C1': {'name': 'C1_ICT_share_PC_EMP', 'dataset': 'isoc_sks_itspt'},
    'C2': {'name': 'C2_Labour_cost_EUR_per_hour', 'dataset': 'lc_lci_lev'},
    'C3': {'name': 'C3_REN_ELC_PC', 'dataset': 'nrg_ind_ren'},
    'C4': {'name': 'C4_Electricity_price_EUR_per_kWh', 'dataset': 'nrg_pc_205'},
    'C5': {'name': 'C5_Population_density_PER_KM2', 'dataset': 'tgs00024'}
}

# ====================================================================
# 2. HELPER FUNCTIONS
# ====================================================================
def get_country_code(region_code):
    """Maps NUTS-2 to NUTS-0, specifically handling EL for Greece."""
    region_code = str(region_code).strip().upper()
    if region_code.startswith('EL'):
        return 'EL'
    return region_code[:2]

def get_target_regions(region_set, custom_text=None):
    if region_set == "IJGI_24_fixed":
        return LOCKED_24_REGIONS.copy()
    else:
        if not custom_text or not custom_text.strip():
            raise ValueError("Custom region list is empty.")
        # Parse comma or newline separated
        regions = re.split(r'[,\n]+', custom_text)
        return [r.strip().upper() for r in regions if r.strip()]

def get_empty_nan_df(target_regions, criterion, year):
    """Returns a dataframe populated with NaNs for diagnostic purposes."""
    df = pd.DataFrame({'region': target_regions})
    df['country'] = df['region'].apply(get_country_code)
    df['value'] = np.nan
    return format_output(df, criterion, 'diagnostic_missing', year)

# ====================================================================
# 3. UNIVERSAL AUDIT & YEAR DETECTION LOGIC
# ====================================================================
def run_universal_audit():
    """
    Universal Audit Module
    Scans all candidate years, performs methodological filtering,
    and classifies the status of each Year-Criterion combination.
    """
    print("Running universal data audit (2014-Present)... Please wait. This may take a minute.")
    current_year = datetime.now().year
    candidate_years = [str(y) for y in range(2014, current_year + 1)]

    audit_results = []
    complete_years = []
    incomplete_years = []

    # Helper to map regions to required targets
    target_nuts2 = set(LOCKED_24_REGIONS)
    target_nuts0 = set(REQUIRED_NUTS0)

    # Cache raw datasets to speed up the loop
    raw_datasets = {}
    for cx, meta in CRITERIA_META.items():
        try:
            raw_datasets[cx] = eurostat.get_data_df(meta['dataset'])
        except Exception:
            raw_datasets[cx] = None

    for year in candidate_years:
        year_is_complete = True

        for cx in ['C1', 'C2', 'C3', 'C4', 'C5']:
            df_raw = raw_datasets[cx]
            status = ""
            missing_count = 0
            missing_list = []

            if df_raw is None:
                status = "DATASET FETCH FAILED"
                year_is_complete = False
            else:
                geo_col = next((c for c in df_raw.columns if 'geo' in c.lower()), 'geo')

                # Check column existence (C4 requires semesters)
                if cx == 'C4':
                    col_exists = f"{year}-S1" in df_raw.columns and f"{year}-S2" in df_raw.columns
                else:
                    col_exists = year in df_raw.columns

                if not col_exists:
                    status = "YEAR COLUMN MISSING"
                    year_is_complete = False
                    if int(year) >= current_year:
                         status += " (Current year / provisional / expected incomplete)"
                else:
                    # Apply Exact Methodological Filters
                    if cx == 'C1':
                        df_f = df_raw[df_raw['unit'] == 'PC_EMP'].copy()
                    elif cx == 'C2':
                        df_f = df_raw[(df_raw['unit'] == 'EUR') & (df_raw['lcstruct'] == 'D1_D4_MD5') & (df_raw['nace_r2'] == 'B-S_X_O')].copy()
                    elif cx == 'C3':
                        df_f = df_raw[(df_raw['nrg_bal'] == 'REN_ELC') & (df_raw['unit'] == 'PC')].copy()
                    elif cx == 'C4':
                        df_f = df_raw[(df_raw['siec'] == 'E7000') & (df_raw['nrg_cons'] == 'MWH500-1999') & (df_raw['tax'] == 'X_VAT') & (df_raw['currency'] == 'EUR') & (df_raw['unit'] == 'KWH')].copy()
                    elif cx == 'C5':
                        df_f = df_raw[df_raw['unit'] == 'PER_KM2'].copy()

                    if len(df_f) == 0:
                        status = "FILTER / STRUCTURE MISMATCH SUSPECTED"
                        year_is_complete = False
                    else:
                        # Identify missing targets
                        if cx in ['C1', 'C2', 'C3', 'C4']:
                            # National level check
                            countries_present = df_f[geo_col].unique()
                            missing_targets = target_nuts0 - set(countries_present)

                            if not missing_targets:
                                # Check NaNs in the values
                                if cx == 'C4':
                                    df_f['val'] = df_f[[f"{year}-S1", f"{year}-S2"]].apply(pd.to_numeric, errors='coerce').mean(axis=1)
                                else:
                                    df_f['val'] = pd.to_numeric(df_f[year], errors='coerce')

                                df_target = df_f[df_f[geo_col].isin(target_nuts0)]
                                nan_countries = df_target[df_target['val'].isna()][geo_col].tolist()
                                missing_targets = set(nan_countries)

                            if len(missing_targets) == len(target_nuts0):
                                status = "SOURCE GAP SUSPECTED (All target values NaN)"
                                year_is_complete = False
                            elif len(missing_targets) > 0:
                                status = "PARTIAL SOURCE GAP SUSPECTED"
                                year_is_complete = False
                            else:
                                status = "COMPLETE"

                            missing_list = list(missing_targets)

                        elif cx == 'C5':
                            # Regional level check
                            regions_present = df_f[geo_col].unique()
                            missing_targets = target_nuts2 - set(regions_present)

                            if not missing_targets:
                                df_f['val'] = pd.to_numeric(df_f[year], errors='coerce')
                                df_target = df_f[df_f[geo_col].isin(target_nuts2)]
                                nan_regions = df_target[df_target['val'].isna()][geo_col].tolist()
                                missing_targets = set(nan_regions)

                            if len(missing_targets) == len(target_nuts2):
                                status = "SOURCE GAP SUSPECTED (All target values NaN)"
                                year_is_complete = False
                            elif len(missing_targets) > 0:
                                status = "PARTIAL SOURCE GAP SUSPECTED"
                                year_is_complete = False
                            else:
                                status = "COMPLETE"

                            missing_list = list(missing_targets)

            missing_count = len(missing_list)
            audit_results.append({
                'Year': year,
                'Criterion': cx,
                'Status': status,
                'Missing_Targets_Count': missing_count,
                'Missing_Targets_List': ", ".join(missing_list) if missing_list else "None"
            })

        if year_is_complete:
            complete_years.append(year)
        else:
            incomplete_years.append(year)

    df_audit = pd.DataFrame(audit_results)
    return sorted(complete_years, reverse=True), sorted(incomplete_years, reverse=True), df_audit

# ====================================================================
# 4. DATA FETCHING & QA FUNCTIONS (STRICT ENGINE)
# ====================================================================
def format_output(df, criterion, proxy, year):
    df['criterion'] = criterion
    df['dataset'] = CRITERIA_META[criterion]['dataset']
    df['year'] = year
    df['proxy_level'] = proxy
    df['notes'] = "Methodology strictly applied." if proxy != 'diagnostic_missing' else "Missing data (Diagnostic)"
    return df[['region', 'country', 'value', 'criterion', 'dataset', 'year', 'proxy_level', 'notes']]

def apply_national_proxy(df_raw, target_regions, value_col, criterion, year, strict=True):
    df_raw = df_raw.rename(columns={value_col: 'value'})
    df_target = pd.DataFrame({'region': target_regions})
    df_target['country'] = df_target['region'].apply(get_country_code)

    geo_col = next(c for c in df_raw.columns if 'geo' in c.lower())
    needed_countries = df_target['country'].unique()
    df_nat = df_raw[df_raw[geo_col].isin(needed_countries)].copy()

    if df_nat[geo_col].duplicated().any():
        if strict: raise ValueError(f"QA Failed ({criterion}): Multiple records found for the same country.")

    df_merged = pd.merge(df_target, df_nat[[geo_col, 'value']], left_on='country', right_on=geo_col, how='left')

    if df_merged['value'].isna().any():
        missing_regs = df_merged[df_merged['value'].isna()]['region'].tolist()
        if strict: raise ValueError(f"QA Failed ({criterion}): Missing data for regions {missing_regs} in year {year}.")

    df_merged['value'] = pd.to_numeric(df_merged['value'], errors='coerce')
    if strict and df_merged['value'].isna().any():
        raise ValueError(f"QA Failed ({criterion}): Non-numeric values detected.")

    return format_output(df_merged, criterion, "national_proxy", year)

def fetch_c1(year, target_regions, strict=True):
    df = eurostat.get_data_df(CRITERIA_META['C1']['dataset'])
    if year not in df.columns:
        if strict: raise KeyError(f"Year {year} missing in C1.")
        else: return get_empty_nan_df(target_regions, 'C1', year)
    df_f = df[(df['unit'] == 'PC_EMP')].copy()
    return apply_national_proxy(df_f, target_regions, year, 'C1', year, strict)

def fetch_c2(year, target_regions, strict=True):
    df = eurostat.get_data_df(CRITERIA_META['C2']['dataset'])
    if year not in df.columns:
        if strict: raise KeyError(f"Year {year} missing in C2.")
        else: return get_empty_nan_df(target_regions, 'C2', year)

    df_f = df[(df['unit'] == 'EUR') & (df['lcstruct'] == 'D1_D4_MD5') & (df['nace_r2'] == 'B-S_X_O')].copy()
    return apply_national_proxy(df_f, target_regions, year, 'C2', year, strict)

def fetch_c3(year, target_regions, strict=True):
    df = eurostat.get_data_df(CRITERIA_META['C3']['dataset'])
    if year not in df.columns:
        if strict: raise KeyError(f"Year {year} missing in C3.")
        else: return get_empty_nan_df(target_regions, 'C3', year)
    df_f = df[(df['nrg_bal'] == 'REN_ELC') & (df['unit'] == 'PC')].copy()
    return apply_national_proxy(df_f, target_regions, year, 'C3', year, strict)

def fetch_c4(year, target_regions, strict=True):
    df = eurostat.get_data_df(CRITERIA_META['C4']['dataset'])
    s1, s2 = f"{year}-S1", f"{year}-S2"
    if s1 not in df.columns or s2 not in df.columns:
        if strict: raise KeyError(f"Semesters missing for {year} in C4.")
        else: return get_empty_nan_df(target_regions, 'C4', year)

    df_f = df[(df['siec'] == 'E7000') & (df['nrg_cons'] == 'MWH500-1999') & (df['tax'] == 'X_VAT') & (df['currency'] == 'EUR') & (df['unit'] == 'KWH')].copy()
    df_f[year] = df_f[[s1, s2]].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    return apply_national_proxy(df_f, target_regions, year, 'C4', year, strict)

def fetch_c5(year, target_regions, strict=True):
    df = eurostat.get_data_df(CRITERIA_META['C5']['dataset'])
    if year not in df.columns:
        if strict: raise KeyError(f"Year {year} missing in C5.")
        else: return get_empty_nan_df(target_regions, 'C5', year)

    df_f = df[df['unit'] == 'PER_KM2'].copy()
    geo_col = next(c for c in df_f.columns if 'geo' in c.lower())

    df_f = df_f[df_f[geo_col].isin(target_regions)].copy()

    df_target = pd.DataFrame({'region': target_regions})
    df_target['country'] = df_target['region'].apply(get_country_code)

    df_merged = pd.merge(df_target, df_f[[geo_col, year]], left_on='region', right_on=geo_col, how='left')
    df_merged = df_merged.rename(columns={year: 'value'})

    if df_merged['value'].isna().any():
        missing_regs = df_merged[df_merged['value'].isna()]['region'].tolist()
        if strict: raise ValueError(f"QA Failed (C5): Missing data for regions {missing_regs} in year {year}.")

    df_merged['value'] = pd.to_numeric(df_merged['value'], errors='coerce')
    if strict and df_merged['value'].isna().any():
        raise ValueError(f"QA Failed (C5): Non-numeric values detected.")

    return format_output(df_merged, 'C5', "direct_nuts2", year)

# ====================================================================
# 5. ASSEMBLY AND EXPORT
# ====================================================================
def assemble_matrix(year, target_regions, fetch_results, strict=True):
    df_final = pd.DataFrame({'region': target_regions})
    df_final['country'] = df_final['region'].apply(get_country_code)

    for cx in ['C1', 'C2', 'C3', 'C4', 'C5']:
        df_cx = fetch_results[cx]
        col_name = f"{CRITERIA_META[cx]['name']}_{year}"
        df_merge = df_cx[['region', 'value']].rename(columns={'value': col_name})
        df_final = pd.merge(df_final, df_merge, on='region', how='left')

    if len(df_final) != len(target_regions):
        raise ValueError("QA Final Failed: Row count mismatch.")
    if df_final['region'].duplicated().any():
        raise ValueError("QA Final Failed: Duplicated regions.")
    if strict and df_final.isna().any().any():
        raise ValueError("QA Final Failed: NaNs present in the final dashboard-ready matrix.")

    return df_final

# ====================================================================
# 6. UI & MAIN PIPELINE
# ====================================================================
out_log = widgets.Output()
out_dl = widgets.Output()
out_audit_table = widgets.Output()

# UI Controls
region_set_dropdown = widgets.Dropdown(
    options=[('Fixed IJGI 24 regions', 'IJGI_24_fixed'), ('Custom NUTS-2 list', 'custom_list')],
    value='IJGI_24_fixed',
    description='Region set:',
    style={'description_width': 'initial'}
)
run_type_dropdown = widgets.Dropdown(
    options=[('Dashboard-ready (complete data only)', 'dashboard'), ('Diagnostic / exploratory (incomplete data allowed)', 'diagnostic')],
    value='dashboard',
    description='Run type:',
    style={'description_width': 'initial'}
)
year_dropdown = widgets.Dropdown(description='Year:', style={'description_width': 'initial'})
custom_text = widgets.Textarea(
    placeholder='Enter NUTS-2 codes, separated by commas or newlines...',
    description='Custom NUTS-2 list:',
    disabled=True,
    style={'description_width': 'initial'}
)
btn_run = widgets.Button(description='Run Extraction', button_style='primary', disabled=True)

GLOBAL_COMPLETE_YEARS = []
GLOBAL_INCOMPLETE_YEARS = []
GLOBAL_AUDIT_DF = None

def update_ui_state(*args):
    if region_set_dropdown.value == 'custom_list':
        custom_text.disabled = False
    else:
        custom_text.disabled = True

    current_year_val = year_dropdown.value
    if run_type_dropdown.value == 'dashboard':
        year_dropdown.options = [(str(y), str(y)) for y in GLOBAL_COMPLETE_YEARS]
        if current_year_val in GLOBAL_COMPLETE_YEARS:
            year_dropdown.value = current_year_val
        elif '2022' in GLOBAL_COMPLETE_YEARS:
            year_dropdown.value = '2022'
    else:
        all_years = sorted(GLOBAL_COMPLETE_YEARS + GLOBAL_INCOMPLETE_YEARS, reverse=True)
        year_dropdown.options = [(f"{y} (Complete)" if y in GLOBAL_COMPLETE_YEARS else f"{y} (Diagnostic)", str(y)) for y in all_years]
        if current_year_val:
            year_dropdown.value = current_year_val

region_set_dropdown.observe(update_ui_state, names='value')
run_type_dropdown.observe(update_ui_state, names='value')

def run_full_pipeline(b):
    with out_log:
        clear_output()
        with out_dl: clear_output()

        try:
            region_set = region_set_dropdown.value
            run_type = run_type_dropdown.value
            year = year_dropdown.value
            txt = custom_text.value
            strict_mode = (run_type == 'dashboard')

            print(f"--- STARTING EXTRACTION PIPELINE ---")
            print(f"Run Type: {run_type.upper()} | Strict QA: {strict_mode} | Year: {year}")

            target_regions = get_target_regions(region_set, txt)
            print(f"Target Regions ({len(target_regions)}): {', '.join(target_regions[:5])}...")

            fetch_results = {}
            for cx, func in zip(['C1', 'C2', 'C3', 'C4', 'C5'], [fetch_c1, fetch_c2, fetch_c3, fetch_c4, fetch_c5]):
                print(f"Fetching {cx}...")
                fetch_results[cx] = func(year, target_regions, strict=strict_mode)

            print("Assembling final matrix...")
            df_final = assemble_matrix(year, target_regions, fetch_results, strict=strict_mode)

            has_nans = df_final.isna().any().any()

            if has_nans:
                missing_info = []
                for col in df_final.columns[2:]:
                    missing_regs = df_final[df_final[col].isna()]['region'].tolist()
                    if missing_regs:
                        missing_info.append(f"<li><b>{col}</b>: {', '.join(missing_regs)}</li>")

                missing_html = "<ul>" + "".join(missing_info) + "</ul>"

                # BANNER: Warning
                display(HTML(f"""
                <div style='padding: 15px; background-color: #fcf8e3; color: #8a6d3b; border-left: 5px solid #d9534f; border-radius: 4px; margin-bottom: 15px;'>
                    <h4 style='margin-top:0;'>⚠️ Diagnostic Run Completed with Missing Values (NAs)</h4>
                    <p><b>Explicit Warning:</b> This file is for inspection only and must not be used for downstream MCDA analysis with complete-data requirements.</p>
                    <p><b>Missing data details:</b></p>
                    {missing_html}
                </div>
                """))

                csv_filename = f"Diagnostic_Matrix_{len(target_regions)}x5_Eurostat_{year}.csv"
                xlsx_filename = f"Diagnostic_Matrix_{len(target_regions)}x5_Eurostat_{year}_master.xlsx"
                btn_color_1, btn_color_2 = "#f0ad4e", "#d9534f"
                csv_label = "📥 Download Diagnostic CSV"
                xls_label = "📥 Download Diagnostic XLSX"

                # File Generation (Bytes & B64)
                excel_io = io.BytesIO()
                with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:
                    df_final.to_excel(writer, sheet_name='Matrix', index=False)
                    pd.DataFrame([{'Region_Set': region_set, 'Run_Type': run_type, 'Year': year, 'Regions': len(target_regions)}]).to_excel(writer, sheet_name='Config', index=False)
                    for cx in ['C1', 'C2', 'C3', 'C4', 'C5']:
                        fetch_results[cx].to_excel(writer, sheet_name=f"QA_{cx}", index=False)

                csv_b64 = base64.b64encode(df_final.to_csv(index=False).encode()).decode()
                xlsx_b64 = base64.b64encode(excel_io.getvalue()).decode()

                # DOWNLOAD BUTTONS (High visual flow)
                display(HTML(f"<h3>Downloads:</h3>"))
                display(HTML("<p style='color:#d9534f; font-weight:bold;'>This file is for inspection only and must not be used for downstream MCDA analysis with complete-data requirements.</p>"))
                display(HTML(f"""
                <a download="{csv_filename}" href="data:text/csv;base64,{csv_b64}"
                   style="background-color: {btn_color_1}; color: white; padding: 10px 15px; margin: 5px; text-decoration: none; border-radius: 4px; font-weight: bold;">
                   {csv_label}
                </a>
                <a download="{xlsx_filename}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{xlsx_b64}"
                   style="background-color: {btn_color_2}; color: white; padding: 10px 15px; margin: 5px; text-decoration: none; border-radius: 4px; font-weight: bold;">
                   {xls_label}
                </a><br><br>
                """))

                # PREVIEW INTRO
                display(HTML("<p><b>Preview below shows the matrix generated for this run. Missing values are marked as NA (Missing).</b></p>"))

                # MATRIX PREVIEW
                df_preview = df_final.copy().fillna('NA (Missing)')

                def color_na(val):
                    return 'background-color: #ffcccc; color: #a94442; font-weight: bold;' if str(val) == 'NA (Missing)' else ''

                if hasattr(df_preview.style, 'map'):
                    styled_df = df_preview.style.map(color_na)
                    styled_head = df_preview.head(50).style.map(color_na)
                else:
                    styled_df = df_preview.style.applymap(color_na)
                    styled_head = df_preview.head(50).style.applymap(color_na)

                if region_set == 'IJGI_24_fixed' or len(df_preview) <= 50:
                    display(styled_df)
                else:
                    display(HTML("<p><i>Preview limited to the first 50 rows. The downloaded file contains the full matrix.</i></p>"))
                    display(styled_head)

            else:
                # BANNER: Success
                display(HTML("""
                <div style='padding: 15px; background-color: #dff0d8; color: #3c763d; border-left: 5px solid #4CAF50; border-radius: 4px; margin-bottom: 15px;'>
                    <h4 style='margin-top:0;'>✅ Extraction completed successfully</h4>
                    The generated CSV file can be dowloaded below, it contains complete data and is ready for downstream MCDA analysis.
                </div>
                """))

                csv_filename = f"Decision_Matrix_{len(target_regions)}x5_Eurostat_{year}.csv"
                xlsx_filename = f"Decision_Matrix_{len(target_regions)}x5_Eurostat_{year}_master.xlsx"
                btn_color_1, btn_color_2 = "#4CAF50", "#2196F3"
                csv_label = "📥 Download CSV (For Dashboard)"
                xls_label = "📥 Download Master XLSX"

                # File Generation (Bytes & B64)
                excel_io = io.BytesIO()
                with pd.ExcelWriter(excel_io, engine='openpyxl') as writer:
                    df_final.to_excel(writer, sheet_name='Matrix', index=False)
                    pd.DataFrame([{'Region_Set': region_set, 'Run_Type': run_type, 'Year': year, 'Regions': len(target_regions)}]).to_excel(writer, sheet_name='Config', index=False)
                    for cx in ['C1', 'C2', 'C3', 'C4', 'C5']:
                        fetch_results[cx].to_excel(writer, sheet_name=f"QA_{cx}", index=False)

                csv_b64 = base64.b64encode(df_final.to_csv(index=False).encode()).decode()
                xlsx_b64 = base64.b64encode(excel_io.getvalue()).decode()

                # DOWNLOAD BUTTONS (High visual flow)
                display(HTML(f"<h3>Downloads:</h3>"))
                display(HTML(f"""
                <a download="{csv_filename}" href="data:text/csv;base64,{csv_b64}"
                   style="background-color: {btn_color_1}; color: white; padding: 10px 15px; margin: 5px; text-decoration: none; border-radius: 4px; font-weight: bold;">
                   {csv_label}
                </a>
                <a download="{xlsx_filename}" href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{xlsx_b64}"
                   style="background-color: {btn_color_2}; color: white; padding: 10px 15px; margin: 5px; text-decoration: none; border-radius: 4px; font-weight: bold;">
                   {xls_label}
                </a><br><br>
                """))

                # PREVIEW INTRO
                display(HTML("<p><b>Preview below shows the matrix generated for this run.</b></p>"))

                # MATRIX PREVIEW
                if region_set == 'IJGI_24_fixed' or len(df_final) <= 50:
                    display(df_final)
                else:
                    display(HTML("<p><i>Preview limited to the first 50 rows. The downloaded file contains the full matrix.</i></p>"))
                    display(df_final.head(50))

        except Exception as e:
            print(f"\n❌ PIPELINE HALTED: {str(e)}")

btn_run.on_click(run_full_pipeline)

# ====================================================================
# 7. INIT SCRIPT & UI PRESENTATION
# ====================================================================
ui_html = """
<h2>Eurostat Extraction Pipeline (Upstream to MCDA Dashboard)</h2>
<p style="font-size:14px; color:#555;">
    <b>This notebook builds a Eurostat-based decision matrix for the MCDA dashboard.</b> It retrieves and validates five criteria, applies national proxies where needed, checks data quality and exports a dashboard-ready CSV plus a QA workbook.
</p>
<p style="font-size:14px; color:#31708f; background-color:#d9edf7; padding: 10px; border-radius: 4px; border-left: 4px solid #31708f;">
    <b>Dashboard-ready years</b> are only the years with COMPLETE status for all five criteria.<br>
    <b>Diagnostic years</b> may contain missing values caused either by real source gaps or by filter/dataset-structure mismatches. See the audit summary below.<br>
    <i>Selecting another valid year does not require exploratory mode. Diagnostic/exploratory mode is only for testing custom regions or inspecting incomplete data.</i>
</p>
<p style="font-size:13px; color:#8a6d3b; background-color:#fcf8e3; padding: 10px; border-radius: 4px; border-left: 4px solid #faebcc;">
    Disclaimer: the extraction logic and data-availability checks implemented in this notebook were cross-checked against Eurostat and are considered correct at the time of use. Missing data were also manually verified in April 2026 in the Eurostat Data Browser: for C2 / lc_lci_lev, years 2014, 2015, 2017, 2018 and 2019 were confirmed as unavailable under the applied study filters, and for C5 / tgs00024, PT17 was confirmed as missing for 2023 and 2024.
</p>

<div style="display: flex; gap: 20px; margin-bottom: 20px; flex-wrap: wrap;">
    <div style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #4CAF50; flex: 1; min-width: 300px;">
        <h4 style="margin-top: 0;">How to use</h4>
        <ol style="margin-top: 5px; margin-bottom: 0; padding-left: 20px;">
            <li>Choose the <b>Region set</b></li>
            <li>Choose the <b>Run type</b></li>
            <li>Choose the <b>Year</b> (choice is independent from region choice)</li>
            <li>Run extraction</li>
            <li>Download the generated outputs</li>
        </ol>
    </div>

    <div style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #2196F3; flex: 1; min-width: 300px;">
        <h4 style="margin-top: 0;">Glossary</h4>
        <ul style="margin-top: 5px; margin-bottom: 0; font-size: 13px; padding-left: 20px;">
            <li><b>Fixed IJGI 24 regions</b> = fixed study region set</li>
            <li><b>Custom NUTS-2 list</b> = user-defined regional codes</li>
            <li><b>Dashboard-ready</b> = strictly enforces complete data only (no missing values)</li>
            <li><b>Diagnostic / exploratory</b> = allows incomplete years and/or custom region experiments</li>
            <li><b>National proxy</b> = the same national value is assigned to all selected regions from the same country</li>
        </ul>
    </div>
</div>
"""
display(HTML(ui_html))

try:
    complete_y, incomplete_y, audit_df = run_universal_audit()
    GLOBAL_COMPLETE_YEARS = complete_y
    GLOBAL_INCOMPLETE_YEARS = incomplete_y
    GLOBAL_AUDIT_DF = audit_df

    if not GLOBAL_COMPLETE_YEARS and not GLOBAL_INCOMPLETE_YEARS:
        display(HTML("<h4 style='color:red;'>Fatal: No data could be retrieved. Eurostat APIs may have changed.</h4>"))
    else:
        update_ui_state()
        btn_run.disabled = False

        display(widgets.VBox([
            widgets.HBox([region_set_dropdown, custom_text]),
            widgets.HBox([run_type_dropdown, year_dropdown]),
            widgets.HTML("<hr>"),
            btn_run
        ]))
        display(out_log)
        display(out_dl)

        # Display the Universal Audit Table
        with out_audit_table:
            clear_output()

            # Criteria legend right above the audit table
            display(HTML("<p style='font-size:13px; color:#555;'>Criteria legend: C1 = ICT specialists share in employment; C2 = labour cost per hour; C3 = renewable electricity share; C4 = non-household electricity price; C5 = population density.</p>"))

            # We highlight rows that are NOT "COMPLETE" in a subtle red
            def color_audit(row):
                if row['Status'] != 'COMPLETE':
                    return ['background-color: #fde8e8'] * len(row)
                return [''] * len(row)

            styled_audit = audit_df.style.apply(color_audit, axis=1)
            display(styled_audit)

        audit_accordion = widgets.Accordion(children=[out_audit_table])
        audit_accordion.set_title(0, 'Data Availability Audit Summary (2014-2026)')
        display(widgets.HTML("<br><br>"))
        display(audit_accordion)

except Exception as e:
    display(HTML(f"<h4 style='color:red;'>Initialization Error: {str(e)}</h4>"))

