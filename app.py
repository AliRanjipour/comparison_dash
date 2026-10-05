import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Iran Data & Comparison Hub | مرکز داده و مقایسه ایران",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CONSTANTS & COUNTRY GROUPS
# -----------------------------------------------------------------------------
# Iran, Neighbors & The Region (18 ISO3 codes)
REGIONAL_ISO3 = [
    'IRN', 'TUR', 'PAK', 'AFG', 'AZE', 'ARM', 'IRQ', 'KWT', 'QAT',
    'ARE', 'SAU', 'BHR', 'OMN', 'ISR', 'EGY', 'JOR', 'PSE', 'YEM'
]

# Custom Indicator Hierarchy Order
CUSTOM_INDICATOR_ORDER = [
    'eco_gdp_cur',
    'eco_gdp_cap_cur',
    'eco_gdp_gro',
    'eco_inf_rat',
    'eco_emp_une',
    'pop_tot',
    'hea_lif_mf',
    'edu_lit_rat',
    'ene_tot_cap',
    'pol_fre_exp'
]

CUSTOM_CAT0_ORDER_EN = [
    'Economy',
    'Population',
    'Health',
    'Education',
    'Energy',
    'Environment',
    'Politics & Human Rights'
]

CUSTOM_CAT0_ORDER_FA = [
    'اقتصاد',
    'جمعیت',
    'بهداشت و درمان',
    'آموزش',
    'انرژی',
    'محیط زیست',
    'سیاست و حقوق بشر'
]

COLOR_PALETTES = {
    'Viridis': px.colors.sequential.Viridis,
    'Blues': px.colors.sequential.Blues,
    'Greens': px.colors.sequential.Greens,
    'Reds': px.colors.sequential.Reds,
    'YlOrRd': px.colors.sequential.YlOrRd,
    'RdYlGn': px.colors.diverging.RdYlGn,
    'Purples': px.colors.sequential.Purples,
    'Oranges': px.colors.sequential.Oranges
}

def get_indicator_rank(ind_id):
    """Return explicit sorting rank for indicators based on hierarchy."""
    if ind_id in CUSTOM_INDICATOR_ORDER:
        return CUSTOM_INDICATOR_ORDER.index(ind_id)
    return 999

def get_cat0_rank(cat0_name, is_fa=False):
    """Return explicit sorting rank for Main Categories."""
    order_list = CUSTOM_CAT0_ORDER_FA if is_fa else CUSTOM_CAT0_ORDER_EN
    if cat0_name in order_list:
        return order_list.index(cat0_name)
    return 999

# -----------------------------------------------------------------------------
# DATA LOADING (LOCAL CSV ENGINE)
# -----------------------------------------------------------------------------
def find_file(filename):
    """Search candidate directories for local CSV files."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), filename) if '__file__' in globals() else None,
        os.path.join(os.getcwd(), filename),
        filename
    ]
    for c in candidates:
        if c and os.path.exists(c):
            return c
    return None

def build_full_csv_dataset():
    """Build dataset containing ALL indicators from local master files + data.csv if available."""
    m_en_path = find_file('master_en.csv')
    m_fa_path = find_file('master_fa.csv')
    data_path = find_file('data.csv')
    geo_path = find_file('geo.csv')

    m_en = pd.read_csv(m_en_path) if m_en_path else pd.DataFrame()
    m_fa = pd.read_csv(m_fa_path) if m_fa_path else pd.DataFrame()

    if m_en.empty:
        return pd.DataFrame()

    col_rename_en = {
        'category_0': 'category_0_en',
        'category_1': 'category_1_en',
        'category_2': 'category_2_en',
        'category_3': 'category_3_en',
        'category_4': 'category_4_en',
        'unit': 'unit_en',
        'data_source': 'source_en',
        'main_source': 'main_source_en'
    }
    m_en = m_en.rename(columns={k: v for k, v in col_rename_en.items() if k in m_en.columns})
    if 'source_en' not in m_en.columns and 'data_source_en' in m_en.columns:
        m_en['source_en'] = m_en['data_source_en']
    if 'main_source_en' not in m_en.columns:
        m_en['main_source_en'] = m_en.get('source_en', '')

    col_rename_fa = {
        'data_source_fa': 'source_fa',
        'data_source': 'source_fa'
    }
    m_fa = m_fa.rename(columns={k: v for k, v in col_rename_fa.items() if k in m_fa.columns})
    if 'main_source_fa' not in m_fa.columns:
        m_fa['main_source_fa'] = m_fa.get('source_fa', '')

    if not m_fa.empty:
        master = pd.merge(m_en, m_fa, on='id', how='left', suffixes=('', '_fa_dup'))
    else:
        master = m_en.copy()

    for col in ['category_0_fa', 'category_1_fa', 'category_2_fa', 'label_fa', 'unit_fa', 'source_fa', 'main_source_fa', 'detail_fa']:
        if col not in master.columns:
            master[col] = master.get('label_en' if 'label' in col else 'category_0_en', '')

    if data_path and geo_path:
        data_df = pd.read_csv(data_path)
        geo_df = pd.read_csv(geo_path)
        
        if 'time_id' in data_df.columns:
            data_df = data_df.rename(columns={'time_id': 'year'})
            
        merged_data = pd.merge(data_df, geo_df, on='iso3', how='left')
        full_df = pd.merge(master, merged_data, on='id', how='left')
    else:
        full_df = master.copy()
        full_df['iso3'] = np.nan
        full_df['year'] = np.nan
        full_df['value'] = np.nan
        full_df['country_name_en'] = np.nan
        full_df['country_name_fa'] = np.nan
        full_df['latitude'] = np.nan
        full_df['longitude'] = np.nan

    return full_df

@st.cache_data(ttl=3600)
def load_dashboard_data():
    """Load dataset directly from local CSV files."""
    df = build_full_csv_dataset()
    if not df.empty:
        return df
    st.error("Could not load master CSV files. Please ensure master_en.csv, master_fa.csv, data.csv, and geo.csv are in the project folder.")
    return pd.DataFrame()

df = load_dashboard_data()

if df.empty:
    st.error("No data available to render dashboard.")
    st.stop()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & LANGUAGE TOGGLE
# -----------------------------------------------------------------------------
st.sidebar.title("تنظیمات داشبورد" if 'lang_state' in st.session_state and st.session_state['lang_state'] == 'فارسی' else "Dashboard Controls")

lang = st.sidebar.radio(
    "Language / زبان",
    options=["English", "فارسی"],
    index=0,
    horizontal=True,
    key='lang_state'
)

is_fa = (lang == "فارسی")

# Inject CSS for LTR / RTL Alignment
if is_fa:
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                direction: rtl;
                text-align: right;
                font-family: 'Vazirmatn', 'Tahoma', sans-serif;
            }
            .stSidebar {
                direction: rtl;
                text-align: right;
            }
            div[data-testid="stMetricValue"] {
                text-align: right;
            }
        </style>
    """, unsafe_allow_html=True)

# Field Mapper
col_cat0 = 'category_0_fa' if is_fa and 'category_0_fa' in df.columns else 'category_0_en'
col_cat1 = 'category_1_fa' if is_fa and 'category_1_fa' in df.columns else 'category_1_en'
col_cat2 = 'category_2_fa' if is_fa and 'category_2_fa' in df.columns else 'category_2_en'
col_label = 'label_fa' if is_fa and 'label_fa' in df.columns else 'label_en'
col_unit = 'unit_fa' if is_fa and 'unit_fa' in df.columns else 'unit_en'
col_source = 'source_fa' if is_fa and 'source_fa' in df.columns else 'source_en'
col_main_source = 'main_source_fa' if is_fa and 'main_source_fa' in df.columns else 'main_source_en'
col_detail = 'detail_fa' if is_fa and 'detail_fa' in df.columns else 'detail_en'
col_country = 'country_name_fa' if is_fa and 'country_name_fa' in df.columns else 'country_name_en'

st.sidebar.markdown("---")

# 1. Cascading Selection (FILTERED TO AVAILABLE DATA ONLY & CUSTOM SORTED)
df_has_data = df.dropna(subset=['value'])
if df_has_data.empty:
    df_has_data = df.copy()

raw_cat0 = [c for c in df_has_data[col_cat0].dropna().unique() if str(c).strip() != '']
if not raw_cat0:
    raw_cat0 = [c for c in df_has_data['category_0_en'].dropna().unique() if str(c).strip() != '']

cat0_options = sorted(raw_cat0, key=lambda c: get_cat0_rank(c, is_fa))

selected_cat0 = st.sidebar.selectbox(
    ("بخش (Main Category)" if is_fa else "Main Category (بخش)"),
    options=cat0_options
)

df_cat0_data = df_has_data[(df_has_data[col_cat0] == selected_cat0) | (df_has_data['category_0_en'] == selected_cat0)]

raw_cat1 = [c for c in df_cat0_data[col_cat1].dropna().unique() if str(c).strip() != '']
if not raw_cat1:
    raw_cat1 = [c for c in df_cat0_data['category_1_en'].dropna().unique() if str(c).strip() != '']

def get_cat1_rank(cat1_val):
    matching_ids = df_cat0_data[(df_cat0_data[col_cat1] == cat1_val) | (df_cat0_data['category_1_en'] == cat1_val)]['id'].dropna().unique()
    ranks = [get_indicator_rank(i) for i in matching_ids]
    return min(ranks) if ranks else 999

cat1_options = sorted(raw_cat1, key=get_cat1_rank)

selected_cat1 = st.sidebar.selectbox(
    ("زیربخش (Sub-Category)" if is_fa else "Sub-Category (زیربخش)"),
    options=cat1_options
)

df_cat1_data = df_cat0_data[(df_cat0_data[col_cat1] == selected_cat1) | (df_cat0_data['category_1_en'] == selected_cat1)]

indicator_df = df_cat1_data[['id', col_label]].drop_duplicates().dropna(subset=['id']).copy()
indicator_df['rank'] = indicator_df['id'].apply(get_indicator_rank)
indicator_df = indicator_df.sort_values('rank')

indicator_ids_ordered = list(indicator_df['id'])
selected_indicator_id = indicator_ids_ordered[0] if indicator_ids_ordered else 'eco_gdp_cur'

# Full subset for selected indicator
df_ind = df[df['id'] == selected_indicator_id].copy()

# Unit conversion adjustment: GDP (current US$) raw USD -> Million USD
if selected_indicator_id == 'eco_gdp_cur' and 'value' in df_ind.columns:
    df_ind['value'] = df_ind['value'] / 1_000_000.0
indicator_meta = df_ind.iloc[0]

st.sidebar.markdown("---")

# 2. Timeframe Selection (Most Recent Data vs. Specific Year)
st.sidebar.subheader("دوره زمانی" if is_fa else "Timeframe")

year_mode_fa = ["جدیدترین داده‌های موجود", "سال مشخص"]
year_mode_en = ["Most Recent Data", "Specific Year"]
year_mode_options = year_mode_fa if is_fa else year_mode_en

selected_year_mode = st.sidebar.radio(
    "نحوه انتخاب سال" if is_fa else "Timeframe Mode",
    options=year_mode_options,
    index=0
)

is_most_recent = (selected_year_mode in ["Most Recent Data", "جدیدترین داده‌های موجود"])

available_years = sorted([int(y) for y in df_ind['year'].dropna().unique()])

if available_years:
    if is_most_recent:
        selected_year = max(available_years)
        st.sidebar.caption(
            f"جدیدترین سال داده: **{selected_year}**" if is_fa else f"Most recent data year: **{selected_year}**"
        )
        if 'iso3' in df_ind.columns and 'value' in df_ind.columns:
            df_all_year = df_ind.dropna(subset=['value']).sort_values('year').groupby('iso3').last().reset_index()
        else:
            df_all_year = df_ind[df_ind['year'] == selected_year].copy()
    else:
        selected_year = st.sidebar.slider(
            "سال انتخاب‌شده" if is_fa else "Select Year",
            min_value=min(available_years),
            max_value=max(available_years),
            value=max(available_years),
            step=1
        )
        df_all_year = df_ind[df_ind['year'] == selected_year].copy()
else:
    selected_year = 2024
    df_all_year = pd.DataFrame()
    st.sidebar.info("اطلاعات زمانی ثبت نشده است" if is_fa else "No time series data registered for this indicator yet.")

st.sidebar.markdown("---")

# Preset Peer Group Filters
group_options_en = ["Global", "Iran, Neighbors & The Region", "Peer Countries", "Custom"]
group_options_fa = ["جهانی", "ایران، همسایگان و کشورهای منطقه", "کشورهای همتراز", "سفارشی"]
group_options = group_options_fa if is_fa else group_options_en

selected_group_label = st.sidebar.radio(
    "فیلتر گروه‌های کشوری" if is_fa else "Preset Peer Groups",
    options=group_options,
    index=0
)

if selected_group_label in ["Global", "جهانی"]:
    group_type = "Global"
elif selected_group_label in ["Regional", "منطقه‌ای", "Iran, Neighbors & The Region", "ایران، همسایگان و کشورهای منطقه"]:
    group_type = "Regional"
elif selected_group_label in ["Peer Countries", "کشورهای همتراز"]:
    group_type = "Peer Countries"
else:
    group_type = "Custom"

geo_file_path = find_file('geo.csv')
if geo_file_path:
    geo_df_temp = pd.read_csv(geo_file_path)
    all_countries_map = dict(zip(geo_df_temp['iso3'], geo_df_temp[col_country]))
else:
    clean_geo_df = df[['iso3', col_country]].dropna().drop_duplicates()
    all_countries_map = dict(zip(clean_geo_df['iso3'], clean_geo_df[col_country]))

if group_type == "Global":
    active_iso3_list = list(df['iso3'].dropna().unique()) if 'iso3' in df.columns else []

elif group_type == "Regional":
    active_iso3_list = [c for c in REGIONAL_ISO3 if 'iso3' in df.columns and c in df['iso3'].unique()]

elif group_type == "Peer Countries":
    if not df_all_year.empty and 'IRN' in df_all_year['iso3'].dropna().values:
        df_sorted = df_all_year.dropna(subset=['value']).sort_values('value').reset_index(drop=True)
        if 'IRN' in df_sorted['iso3'].values:
            iran_idx = df_sorted[df_sorted['iso3'] == 'IRN'].index[0]
            lower_10 = df_sorted.iloc[max(0, iran_idx - 10):iran_idx]['iso3'].tolist()
            higher_10 = df_sorted.iloc[iran_idx + 1:min(len(df_sorted), iran_idx + 11)]['iso3'].tolist()
            active_iso3_list = lower_10 + ['IRN'] + higher_10
        else:
            active_iso3_list = REGIONAL_ISO3[:10]
    else:
        active_iso3_list = REGIONAL_ISO3[:10]

else:  # Custom
    country_options = sorted(list(all_countries_map.keys()), key=lambda x: str(all_countries_map.get(x, x)))
    active_iso3_list = st.sidebar.multiselect(
        "انتخاب کشورهای سفارشی" if is_fa else "Select Custom Countries",
        options=country_options,
        default=['IRN', 'TUR', 'SAU'],
        format_func=lambda x: all_countries_map.get(x, x),
        key=f"custom_country_multiselect_{lang}"
    )

# Only force 'IRN' into active_iso3_list if NOT in Custom mode
if group_type != "Custom" and 'IRN' not in active_iso3_list:
    active_iso3_list.append('IRN')

# -----------------------------------------------------------------------------
# MAIN WORKSPACE - HEADER BANNER & CONTEXT
# -----------------------------------------------------------------------------
indicator_name = indicator_meta.get(col_label, selected_indicator_id)
unit_str = str(indicator_meta.get(col_unit, '')) if pd.notna(indicator_meta.get(col_unit, '')) else ""
source_str = str(indicator_meta.get(col_source, '')) if pd.notna(indicator_meta.get(col_source, '')) else ""
main_source_str = str(indicator_meta.get(col_main_source, '')) if pd.notna(indicator_meta.get(col_main_source, '')) else ""
detail_str = str(indicator_meta.get(col_detail, '')) if pd.notna(indicator_meta.get(col_detail, '')) else ""
data_url_str = str(indicator_meta.get('data_url', '')) if pd.notna(indicator_meta.get('data_url', '')) else ""

header_year_str = ("جدیدترین داده‌ها" if is_fa else "Most Recent Data") if is_most_recent else str(selected_year)
st.title(f"{indicator_name} ({header_year_str})")

# Metadata Banner
with st.container():
    st.markdown(f"**{'توضیحات شاخص' if is_fa else 'Description'}:** {detail_str}")
    
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.markdown(f"**{'واحد اندازه‌گیری' if is_fa else 'Unit'}:** {unit_str}")
    with m_col2:
        # Combined Data Source & Primary Source
        combined_source = source_str
        if main_source_str and str(main_source_str).strip() != "":
            primary_label = "منبع اصلی" if is_fa else "Primary Source"
            combined_source += f" ({primary_label}: {main_source_str})"
            
        source_link_html = f"<a href='{data_url_str}' target='_blank'>{combined_source}</a>" if data_url_str else combined_source
        st.markdown(f"**{'منبع داده' if is_fa else 'Data Source'}:** {source_link_html}", unsafe_allow_html=True)

st.markdown("---")

# -----------------------------------------------------------------------------
# DYNAMIC RANKING & DATA SLICING
# -----------------------------------------------------------------------------
df_year_active = df_all_year[df_all_year['iso3'].isin(active_iso3_list)].copy() if not df_all_year.empty else pd.DataFrame()

sorting_order = str(indicator_meta.get('sorting_order', 'des')).lower()
is_ascending = (sorting_order == 'asc')

if not df_all_year.empty and 'value' in df_all_year.columns:
    df_all_year['global_rank'] = df_all_year['value'].rank(ascending=is_ascending, method='min')
    df_reg_year = df_all_year[df_all_year['iso3'].isin(REGIONAL_ISO3)].copy()
    df_reg_year['regional_rank'] = df_reg_year['value'].rank(ascending=is_ascending, method='min')
else:
    df_reg_year = pd.DataFrame()

if not df_year_active.empty and 'value' in df_year_active.columns:
    df_year_active['active_rank'] = df_year_active['value'].rank(ascending=is_ascending, method='min')

iran_data_active = df_year_active[df_year_active['iso3'] == 'IRN'] if not df_year_active.empty else pd.DataFrame()
iran_data_global = df_all_year[df_all_year['iso3'] == 'IRN'] if not df_all_year.empty else pd.DataFrame()
iran_data_reg = df_reg_year[df_reg_year['iso3'] == 'IRN'] if not df_reg_year.empty else pd.DataFrame()

# -----------------------------------------------------------------------------
# TOP KPI CARDS ROW (CUSTOM HTML STYLING FOR ENHANCED VISIBILITY & SIZE)
# -----------------------------------------------------------------------------
kpi1, kpi2, kpi3 = st.columns([1, 1, 2])

with kpi1:
    kpi1_title = "مقدار ایران" if is_fa else "Iran Value"
    if not iran_data_active.empty and pd.notna(iran_data_active['value'].values[0]):
        iran_val = iran_data_active['value'].values[0]
        iran_year_val = int(iran_data_active['year'].values[0]) if 'year' in iran_data_active.columns and pd.notna(iran_data_active['year'].values[0]) else selected_year
        val_str = f"{iran_val:,.2f}"
        sub_str = f"سال {iran_year_val}" if is_fa else f"Year {iran_year_val}"
    else:
        val_str = "N/A"
        sub_str = ""

    html_kpi1 = f"""
    <div style="background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="font-size: 18px; font-weight: 700; color: #2C3E50; margin-bottom: 6px;">{kpi1_title}</div>
        <div style="font-size: 28px; font-weight: 800; color: #111;">
            {val_str} <span style="font-size: 13px; font-weight: 500; color: #666; margin-left: 2px;">{unit_str}</span>
        </div>
        <div style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">{sub_str}</div>
    </div>
    """
    st.markdown(html_kpi1, unsafe_allow_html=True)

with kpi2:
    if group_type == "Regional":
        rank_title = "رتبه منطقه‌ای ایران" if is_fa else "Iran Regional Rank"
        if not iran_data_reg.empty and pd.notna(iran_data_reg.get('regional_rank', pd.Series([np.nan])).values[0]):
            r_rank = int(iran_data_reg['regional_rank'].values[0])
            tot_reg = len(df_reg_year.dropna(subset=['value']))
            out_txt = "از" if is_fa else "out of"
            rank_str = f"{r_rank} {out_txt} {tot_reg}"
            sub_str = "کمتر بهتر است" if is_ascending and is_fa else ("Lower is Better" if is_ascending else ("بیشتر بهتر است" if is_fa else "Higher is Better"))
        else:
            rank_str = "N/A"
            sub_str = ""
    else:
        rank_title = "رتبه جهانی ایران" if is_fa else "Iran Global Rank"
        if not iran_data_global.empty and pd.notna(iran_data_global.get('global_rank', pd.Series([np.nan])).values[0]):
            g_rank = int(iran_data_global['global_rank'].values[0])
            tot_glob = len(df_all_year.dropna(subset=['value']))
            out_txt = "از" if is_fa else "out of"
            rank_str = f"{g_rank} {out_txt} {tot_glob}"
            sub_str = "کمتر بهتر است" if is_ascending and is_fa else ("Lower is Better" if is_ascending else ("بیشتر بهتر است" if is_fa else "Higher is Better"))
        else:
            rank_str = "N/A"
            sub_str = ""

    html_kpi2 = f"""
    <div style="background-color: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
        <div style="font-size: 18px; font-weight: 700; color: #2C3E50; margin-bottom: 6px;">{rank_title}</div>
        <div style="font-size: 28px; font-weight: 800; color: #111;">
            {rank_str}
        </div>
        <div style="font-size: 12px; color: #7f8c8d; margin-top: 4px;">{sub_str}</div>
    </div>
    """
    st.markdown(html_kpi2, unsafe_allow_html=True)

with kpi3:
    df_iran_hist = df_ind[(df_ind['iso3'] == 'IRN') & (df_ind['year'] >= 1990)].sort_values('year') if 'iso3' in df_ind.columns and 'year' in df_ind.columns else pd.DataFrame()
    df_iran_hist = df_iran_hist.dropna(subset=['value'])
    
    if not df_iran_hist.empty:
        fig_spark = px.line(
            df_iran_hist, 
            x='year', 
            y='value',
            title="روند تاریخی ایران (۱۹۹۰ تا کنون)" if is_fa else "Iran Historical Trend (1990 - Present)",
            labels={'year': 'Year' if not is_fa else 'سال', 'value': unit_str}
        )
        line_color = '#2E7D32' if not is_ascending else '#C62828'
        fig_spark.update_traces(line_color=line_color, line_width=2.5)
        fig_spark.update_layout(
            height=180, 
            margin=dict(l=15, r=15, t=35, b=15),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='#ECEFF1')
        )
        st.plotly_chart(fig_spark, use_container_width=True)
    else:
        st.info("داده‌های تاریخی برای ایران موجود نیست" if is_fa else "Historical time series for Iran not available.")

st.markdown("---")

# -----------------------------------------------------------------------------
# VISUAL MODULES: MAP & BAR CHART
# -----------------------------------------------------------------------------
palette_name = str(indicator_meta.get('colour_range', 'Viridis'))
color_scale = COLOR_PALETTES.get(palette_name, px.colors.sequential.Viridis)

vis1, vis2 = st.columns([1, 1])

with vis1:
    st.subheader("نقشه پراکندگی جغرافیایی" if is_fa else "Geospatial Map")
    
    df_map = df_year_active.dropna(subset=['value', 'iso3']).copy() if not df_year_active.empty and 'value' in df_year_active.columns else pd.DataFrame()
    
    if not df_map.empty:
        # Determine rank mapping for Map Pop-up (always Global Rank)
        rank_dict_map = dict(zip(df_all_year['iso3'], df_all_year['global_rank']))
        tot_count_map = len(df_all_year.dropna(subset=['value']))
        rank_label_map = "رتبه جهانی" if is_fa else "Global Rank"

        out_txt_map = "از" if is_fa else "out of"

        def get_map_rank_str(iso):
            r = rank_dict_map.get(iso, np.nan)
            if pd.notna(r):
                return f"{int(r)} {out_txt_map} {tot_count_map}"
            return "N/A"

        df_map['rank_str'] = df_map['iso3'].apply(get_map_rank_str)
        df_map['formatted_value'] = df_map['value'].apply(lambda v: f"{v:,.2f}")
        df_map['unit_str_col'] = unit_str
        df_map['year_str_col'] = df_map['year'].fillna(selected_year).astype(int).astype(str)
        country_names_map = df_map[col_country] if col_country in df_map.columns else df_map['iso3']

        customdata = np.stack((
            country_names_map,
            df_map['formatted_value'],
            df_map['unit_str_col'],
            df_map['year_str_col'],
            df_map['rank_str']
        ), axis=-1)

        lbl_val = "مقدار" if is_fa else "Value"
        lbl_year = "سال" if is_fa else "Year"

        fig_map = go.Figure(go.Choropleth(
            locations=df_map['iso3'],
            z=df_map['value'],
            colorscale=color_scale,
            customdata=customdata,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br><br>" +
                f"<b>{lbl_val}:</b> %{{customdata[1]}} %{{customdata[2]}}<br>" +
                f"<b>{rank_label_map}:</b> %{{customdata[4]}}<br>" +
                f"<b>{lbl_year}:</b> %{{customdata[3]}}" +
                "<extra></extra>"
            )
        ))
        fig_map.update_geos(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            fitbounds="locations" if len(active_iso3_list) < 30 else False
        )
        fig_map.update_layout(
            height=420,
            margin=dict(l=0, r=0, t=30, b=0),
            coloraxis_colorbar=dict(title=unit_str)
        )
        st.plotly_chart(fig_map, use_container_width=True)
    else:
        st.info("داده‌های جغرافیایی برای انتخاب فعلی ثبت نشده است." if is_fa else "No map data available for the current selection.")

with vis2:
    st.subheader("نمودار مقایسه‌ای کشورها" if is_fa else "Country Comparison Chart")
    
    df_bar = df_year_active.dropna(subset=['value']).sort_values('value', ascending=False).copy() if not df_year_active.empty and 'value' in df_year_active.columns else pd.DataFrame()
    
    if not df_bar.empty:
        if group_type == "Global" and len(df_bar) > 20:
            top_20 = df_bar.head(20)
            if 'IRN' not in top_20['iso3'].values and 'IRN' in df_bar['iso3'].values:
                iran_row = df_bar[df_bar['iso3'] == 'IRN']
                df_bar = pd.concat([top_20, iran_row]).drop_duplicates()
            else:
                df_bar = top_20

        # Map ranks to bar labels (always Global Rank)
        rank_dict_bar = dict(zip(df_all_year['iso3'], df_all_year['global_rank']))
        rank_prefix_bar = "رتبه جهانی" if is_fa else "Global rank"

        def make_bar_text(row):
            val_fmt = f"{row['value']:,.2f}"
            r = rank_dict_bar.get(row['iso3'], np.nan)
            if pd.notna(r):
                return f"{val_fmt} ({rank_prefix_bar}: {int(r)})"
            return val_fmt

        df_bar['bar_text'] = df_bar.apply(make_bar_text, axis=1)

        bar_colors = ['#FF9800' if iso == 'IRN' else '#455A64' for iso in df_bar['iso3']]
        country_col = col_country if col_country in df_bar.columns else 'iso3'
        
        # Calculate X-axis range with room for outside labels
        val_min = df_bar['value'].min()
        val_max = df_bar['value'].max()
        if pd.notna(val_min) and pd.notna(val_max):
            x_min = val_min * 0.90 if val_min >= 0 else val_min * 1.30
            x_max = val_max * 1.35 if val_max >= 0 else val_max * 0.80
            if x_min == x_max:
                x_min = x_min * 0.9 if x_min != 0 else -1
                x_max = x_max * 1.35 if x_max != 0 else 1
        else:
            x_min, x_max = None, None

        fig_bar = go.Figure(go.Bar(
            x=df_bar['value'],
            y=df_bar[country_col],
            orientation='h',
            marker_color=bar_colors,
            text=df_bar['bar_text'],
            textposition='outside',
            cliponaxis=False
        ))
        
        left_margin = 160 if is_fa else 130
        
        fig_bar.update_layout(
            title=f"{'رتبه‌بندی مقایسه‌ای' if is_fa else 'Ranked Comparison'} ({header_year_str})",
            xaxis_title=unit_str,
            yaxis_title=None,
            height=420,
            margin=dict(l=left_margin, r=50, t=30, b=10),
            yaxis=dict(
                autorange="reversed",
                automargin=True,
                tickfont=dict(size=12)
            )
        )
        if x_min is not None and x_max is not None:
            fig_bar.update_xaxes(range=[x_min, x_max])
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("داده‌های مقایسه‌ای برای انتخاب فعلی ثبت نشده است." if is_fa else "No comparison data available for the current selection.")

# -----------------------------------------------------------------------------
# DATA TABLE VIEW & EXPORT
# -----------------------------------------------------------------------------
with st.expander("مشاهده و دریافت داده‌ها" if is_fa else "View & Export Data Table"):
    if not df_year_active.empty and 'active_rank' in df_year_active.columns:
        display_df = df_year_active[[col_country if col_country in df_year_active.columns else 'iso3', 'iso3', 'year', 'value', 'active_rank']].dropna(subset=['value']).sort_values('active_rank')
        display_df.columns = [
            'کشور' if is_fa else 'Country',
            'ISO3',
            'سال' if is_fa else 'Year',
            f'مقدار ({unit_str})' if is_fa else f'Value ({unit_str})',
            'رتبه' if is_fa else 'Rank'
        ]
        st.dataframe(display_df, use_container_width=True)
        
        csv_bytes = display_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="دانلود فایل CSV" if is_fa else "Download CSV",
            data=csv_bytes,
            file_name=f"{selected_indicator_id}_{selected_year}.csv",
            mime="text/csv"
        )
    else:
        st.info("جدول داده برای این شاخص خالی است." if is_fa else "Data table is empty for this indicator.")
