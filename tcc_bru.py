import streamlit as st
import geemap
import ee
import pandas as pd
import json
import tempfile
import os
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import scipy.stats as stats
from io import BytesIO
import zipfile
import requests

import os
import tempfile
from pathlib import Path

# Configuração para deploy
if not os.path.exists('assets'):
    os.makedirs('assets')

# ===================== FUNÇÕES DE PRECIPITAÇÃO (GPM + CHIRPS) =====================
def get_gpm_precip(start_date, end_date, geometry):
    """
    Retorna precipitação média (mm/h) usando GPM IMERG
    """
    try:
        collection = (ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
                      .filterDate(start_date, end_date)
                      .select("precipitationCal"))
        
        image = collection.mean()
        
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=10000,
            maxPixels=1e9
        )
        result = stats.getInfo()
        return result.get('precipitationCal', None), None
    except Exception as e:
        return None, str(e)

def get_chirps_precip(start_date, end_date, geometry):
    """
    Retorna precipitação acumulada (mm) usando CHIRPS
    """
    try:
        collection = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                      .filterDate(start_date, end_date)
                      .select("precipitation"))
        
        image = collection.sum()
        
        stats = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=5000,
            maxPixels=1e9
        )
        result = stats.getInfo()
        return result.get('precipitation', None), None
    except Exception as e:
        return None, str(e)

def get_precip_series(start_date, end_date, geometry, source='both'):
    """
    Retorna série temporal de precipitação
    source: 'gpm', 'chirps', ou 'both'
    """
    results = {}
    errors = {}
    
    if source in ['gpm', 'both']:
        # Para GPM - dados mensais
        gpm_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_date_obj:
            month_start = current_date.strftime('%Y-%m-%d')
            if current_date.month == 12:
                next_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                next_date = current_date.replace(month=current_date.month + 1, day=1)
            month_end = (next_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            try:
                collection = (ee.ImageCollection("NASA/GPM_L3/IMERG_V06")
                              .filterDate(month_start, month_end)
                              .select("precipitationCal"))
                image = collection.mean()
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=10000,
                    maxPixels=1e9
                )
                value = stats.getInfo().get('precipitationCal', None)
                if value:
                    gpm_data.append({'date': current_date.strftime('%Y-%m'), 'precip': value, 'source': 'GPM'})
            except Exception as e:
                errors['GPM'] = str(e)
            
            current_date = next_date
        
        results['gpm'] = pd.DataFrame(gpm_data) if gpm_data else None
    
    if source in ['chirps', 'both']:
        # Para CHIRPS - dados diários agregados por mês
        chirps_data = []
        current_date = datetime.strptime(start_date, '%Y-%m-%d')
        end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
        
        while current_date <= end_date_obj:
            month_start = current_date.strftime('%Y-%m-%d')
            if current_date.month == 12:
                next_date = current_date.replace(year=current_date.year + 1, month=1, day=1)
            else:
                next_date = current_date.replace(month=current_date.month + 1, day=1)
            month_end = (next_date - pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            
            try:
                collection = (ee.ImageCollection("UCSB-CHG/CHIRPS/DAILY")
                              .filterDate(month_start, month_end)
                              .select("precipitation"))
                image = collection.sum()
                stats = image.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=geometry,
                    scale=5000,
                    maxPixels=1e9
                )
                value = stats.getInfo().get('precipitation', None)
                if value:
                    chirps_data.append({'date': current_date.strftime('%Y-%m'), 'precip': value, 'source': 'CHIRPS'})
            except Exception as e:
                errors['CHIRPS'] = str(e)
            
            current_date = next_date
        
        results['chirps'] = pd.DataFrame(chirps_data) if chirps_data else None
    
    return results, errors if errors else None

# Inicialização do Earth Engine
# ===================== CONFIGURAÇÃO EARTH ENGINE =====================
def initialize_earth_engine():
    try:
        ee.Initialize(project='ee-brunamaiiia')
        st.sidebar.success("✅ Earth Engine inicializado")
        return True
    except Exception as e:
        try:
            if 'EE_SERVICE_ACCOUNT' in st.secrets and 'EE_PRIVATE_KEY' in st.secrets:
                service_account = st.secrets['EE_SERVICE_ACCOUNT']
                private_key = st.secrets['EE_PRIVATE_KEY']
                
                credentials = ee.ServiceAccountCredentials(
                    service_account, 
                    key_data=private_key
                )
                
                ee.Initialize(credentials, project='ee-brunamaiiia')
                st.sidebar.success("✅ Earth Engine inicializado (Service Account)")
                return True
            else:
                st.sidebar.warning("⚠️ Earth Engine sem autenticação")
                return False
        except Exception as e:
            st.sidebar.error(f"❌ Erro: {str(e)}")
            return False

# Inicializar Earth Engine
ee_initialized = initialize_earth_engine()

# Configuração da página
st.set_page_config(
    layout='wide',
    page_title="🌊 Análise de Água + Precipitação - Bacia do Pericumã",
    page_icon="🌊"
)

# ===================== ESTILOS E CSS (mantido seu original) =====================
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #546E7A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #2196F3;
        margin-bottom: 1rem;
    }
    .precip-card {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 1rem;
    }
    .info-box {
        background: #FFF8E1;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FFC107;
        margin: 1rem 0;
    }
    .success-box {
        background: #E8F5E8;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
    }
    .warning-box {
        background: #FFF3E0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
    }
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .map-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# ===================== CABEÇALHO =====================
st.markdown('<h1 class="main-header">WebApp para monitoramento da superfície de água na bacia hidrográfica do rio Pericumã</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Monitoramento da dinâmica de corpos d\'água + análise de precipitação (GPM e CHIRPS)</p>', unsafe_allow_html=True)

# ===================== CONFIGURAÇÃO DA BACIA =====================
PERICUMA_ASSET = 'projects/ee-brunamaiiia/assets/Bacia_Pericuma_ZEE_v2'
bacia = ee.FeatureCollection(PERICUMA_ASSET)
geometry = bacia.geometry()

# ===================== CARREGAR DADOS MAPBIOMAS ÁGUA =====================
water_image = ee.Image('projects/mapbiomas-public/assets/brazil/water/collection4/mapbiomas_brazil_collection4_water_v3')
WATER_VALUE = 1

# ===================== SIDEBAR =====================
with st.sidebar:
    # Logos no topo do sidebar
    col1, col2 = st.columns([1, 1])
    with col1:
        if os.path.exists("assets/lageos.jpeg"):
            st.image("assets/lageos.jpeg", width=80)
    with col2:
        if os.path.exists("assets/brasao-normal.png"):
            st.image("assets/brasao-normal.png", width=80)

    st.markdown("### ⚙️ Configurações de Análise")
    
    # Obter anos disponíveis
    band_names = water_image.bandNames().getInfo()
    years = sorted([int(band.split('_')[1]) for band in band_names if band.startswith('classification_')])
    
    selected_years = st.multiselect(
        '**Selecione o(s) ano(s)**',
        options=years,
        default=[2022, 2010, 2000, 1990, 1985],
        help="Selecione os anos para análise temporal"
    )
    
    st.markdown("---")
    st.markdown("### 🌧️ Configurações de Precipitação")
    
    use_precip = st.checkbox("📊 Incluir análise de precipitação", value=True)
    
    if use_precip:
        precip_source = st.selectbox(
            "Fonte de dados de precipitação",
            ["Ambas (GPM + CHIRPS)", "Apenas GPM", "Apenas CHIRPS"],
            help="GPM: dados desde 2000 | CHIRPS: dados desde 1981"
        )
        
        precip_years = st.multiselect(
            "Anos para análise de precipitação",
            options=years,
            default=[2022, 2020, 2018, 2015],
            help="Selecione os anos para comparar com precipitação"
        )
        
        if not precip_years:
            st.warning("Selecione pelo menos um ano para análise de precipitação")
    
    st.markdown("---")
    st.markdown("### 📊 Opções de Visualização")
    
    show_trendline = st.checkbox("📈 Mostrar linha de tendência", value=True)
    smooth_lines = st.checkbox("🔄 Suavizar linhas do gráfico", value=True)
    
    st.markdown("---")
    st.markdown("### 🔍 Análise de Correlação")
    
    correlation_method = st.selectbox(
        "Método de correlação",
        ["Pearson", "Spearman", "Kendall"],
        help="Selecione o método para cálculo de correlação"
    )
    
    st.markdown("---")
    st.markdown("### 💾 Download de Rasters")
    
    download_mode = st.radio(
        "Modo de download:",
        ["Raster único", "Múltiplos rasters"]
    )
    
    if download_mode == "Raster único":
        download_year = st.selectbox(
            "Ano para download do raster",
            options=years,
            index=len(years)-1
        )
    else:
        download_years = st.multiselect(
            "Selecione os anos para download",
            options=years,
            default=[2022, 2020, 2018]
        )
    
    download_resolution = st.slider(
        "Resolução do raster (metros)",
        min_value=30,
        max_value=500,
        value=100,
        step=10
    )
    
    if download_mode == "Múltiplos rasters":
        create_zip = st.checkbox("📦 Compactar em arquivo ZIP", value=True)
        max_downloads = st.slider("Número máximo de rasters", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    st.markdown("### 📋 Informações da Bacia")
    
    try:
        area_bacia = geometry.area().getInfo() / 1e6
        st.metric("Área total da bacia", f"{area_bacia:,.2f} km²")
    except:
        st.warning("⚠️ Não foi possível calcular a área da bacia.")
    
    st.info("""
    **💡 Sobre os dados:**
    - Fonte: MapBiomas Água Collection 4
    - Classe analisada: Água
    - Precipitação GPM: NASA
    - Precipitação CHIRPS: UCSB
    """)

# ===================== FUNÇÃO DE CÁLCULO =====================
def calculate_water_area(year):
    """Calcula área de água usando o valor correto (1)"""
    band_name = f"classification_{year}"
    
    try:
        water_band = water_image.select(band_name)
        water_mask = water_band.eq(WATER_VALUE)
        
        area_m2 = water_mask.multiply(ee.Image.pixelArea())
        
        area_stats = area_m2.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=geometry,
            scale=30,
            maxPixels=1e13,
            bestEffort=True
        )
        
        area_value = area_stats.get(band_name)
        area_m2_value = area_value.getInfo() if area_value else 0
        area_km2 = area_m2_value / 1e6
        
        return area_km2, water_mask
        
    except Exception as e:
        return 0, None

# ===================== FUNÇÕES PARA DOWNLOAD DE RASTER =====================
def download_raster_bacia(year, resolution, geometry):
    """Baixa o raster da bacia para o ano selecionado"""
    try:
        band_name = f"classification_{year}"
        water_band = water_image.select(band_name)
        
        raster_clipped = water_band.clip(geometry)
        
        export_params = {
            'scale': resolution,
            'region': geometry,
            'fileFormat': 'GeoTIFF',
            'formatOptions': {'cloudOptimized': True}
        }
        
        download_url = raster_clipped.getDownloadUrl(export_params)
        return download_url, None
        
    except Exception as e:
        return None, str(e)

def download_multiple_rasters(years, resolution, geometry, max_downloads=5):
    """Baixa múltiplos rasters"""
    try:
        if len(years) > max_downloads:
            return None, f"Selecione no máximo {max_downloads} anos"
        
        download_urls = {}
        for year in years:
            download_url, error = download_raster_bacia(year, resolution, geometry)
            if download_url:
                download_urls[year] = download_url
            else:
                return None, f"Erro no ano {year}: {error}"
        
        return download_urls, None
        
    except Exception as e:
        return None, str(e)

def create_zip_from_urls(download_urls):
    """Cria ZIP a partir das URLs"""
    try:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for year, url in download_urls.items():
                response = requests.get(url)
                if response.status_code == 200:
                    filename = f"agua_pericuma_{year}.tif"
                    zip_file.writestr(filename, response.content)
                else:
                    return None, f"Erro no ano {year}"
        zip_buffer.seek(0)
        return zip_buffer, None
    except Exception as e:
        return None, str(e)

# ===================== FUNÇÃO DE ANÁLISE DE CORRELAÇÃO =====================
def analyze_correlation(df, method='Pearson'):
    """Realiza análise de correlação entre ano e área de água"""
    try:
        if len(df) < 2:
            return None, "Dados insuficientes"
        
        x = df['Ano'].values
        y = df['Área de Água (km²)'].values
        
        if method == 'Pearson':
            corr_coef, p_value = stats.pearsonr(x, y)
        elif method == 'Spearman':
            corr_coef, p_value = stats.spearmanr(x, y)
        elif method == 'Kendall':
            corr_coef, p_value = stats.kendalltau(x, y)
        else:
            corr_coef, p_value = stats.pearsonr(x, y)
        
        abs_corr = abs(corr_coef)
        if abs_corr >= 0.7:
            strength = "Forte"
        elif abs_corr >= 0.3:
            strength = "Moderada"
        else:
            strength = "Fraca"
        
        direction = "positiva" if corr_coef > 0 else "negativa"
        slope, intercept = np.polyfit(x, y, 1)
        trend = "crescente" if slope > 0 else "decrescente"
        
        correlation_result = {
            'coeficiente': corr_coef,
            'p_valor': p_value,
            'força': strength,
            'direção': direction,
            'tendência': trend,
            'inclinação': slope
        }
        
        return correlation_result, None
        
    except Exception as e:
        return None, str(e)

# ===================== MAPA INTERATIVO =====================
st.markdown("## Mapa Interativo")

with st.spinner("Carregando mapa..."):

    import geemap.foliumap as geemap

    m = geemap.Map(center=[-2.8, -44.4], zoom=9)

    study_area = ee.FeatureCollection([ee.Feature(geometry)])

    m.addLayer(study_area, {
        'color': '#E53935',
        'fillColor': '00000000',
        'width': 3
    }, 'Bacia do Pericumã')

    # Adicionar camadas de água para anos selecionados
    if selected_years:
        for year in selected_years:
            area_km2, water_mask = calculate_water_area(year)
            if water_mask:
                water_clipped = water_mask.clip(geometry)
                m.addLayer(
                    water_clipped,
                    {'palette': ['00000000', '#1976D2'], 'min': 0, 'max': 1},
                    f"Água {year} ({area_km2:.1f} km²)"
                )

    m.add_basemap('OpenStreetMap')
    m.add_basemap('Google Terrain')
    m.add_basemap('Google Satellite')
    m.add_layer_control()

    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    m.to_streamlit(height=500)
    st.markdown('</div>', unsafe_allow_html=True)

# ===================== CÁLCULO DAS ÁREAS =====================
if selected_years:
    with st.spinner('Calculando áreas de água...'):
        stats_data = []
        
        for year in selected_years:
            area_km2, _ = calculate_water_area(year)
            total_area_km2 = geometry.area().getInfo() / 1e6
            water_percentage = (area_km2 / total_area_km2) * 100 if total_area_km2 > 0 else 0
            
            stats_data.append({
                "Ano": year,
                "Área de Água (km²)": area_km2,
                "Área Total da Bacia (km²)": total_area_km2,
                "Percentual de Água (%)": water_percentage
            })
    
    df = pd.DataFrame(stats_data).sort_values('Ano')
    
    if df["Área de Água (km²)"].sum() > 0:
        
        # ===================== SEÇÃO DE PRECIPITAÇÃO (NOVA) =====================
        if use_precip and ee_initialized:
            st.markdown("## 🌧️ Análise de Precipitação (GPM + CHIRPS)")
            
            # Determinar fonte baseada na seleção
            source_map = {
                "Ambas (GPM + CHIRPS)": "both",
                "Apenas GPM": "gpm",
                "Apenas CHIRPS": "chirps"
            }
            precip_source_code = source_map.get(precip_source, "both")
            
            # Definir período baseado nos anos selecionados
            if precip_years:
                start_date_precip = f"{min(precip_years)}-01-01"
                end_date_precip = f"{max(precip_years)}-12-31"
                
                with st.spinner("Carregando dados de precipitação..."):
                    precip_results, precip_errors = get_precip_series(
                        start_date_precip, end_date_precip, geometry, precip_source_code
                    )
                
                # Criar DataFrame combinado
                dfs_to_plot = []
                if precip_results.get('gpm') is not None:
                    dfs_to_plot.append(precip_results['gpm'])
                if precip_results.get('chirps') is not None:
                    dfs_to_plot.append(precip_results['chirps'])
                
                if dfs_to_plot:
                    precip_df = pd.concat(dfs_to_plot, ignore_index=True)
                    
                    # Gráfico de precipitação
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        fig_precip = px.bar(
                            precip_df, 
                            x='date', 
                            y='precip', 
                            color='source',
                            title="Precipitação Mensal na Bacia",
                            labels={'date': 'Mês/Ano', 'precip': 'Precipitação (mm)', 'source': 'Fonte'},
                            barmode='group',
                            color_discrete_map={'GPM': '#2196F3', 'CHIRPS': '#4CAF50'}
                        )
                        fig_precip.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            height=400,
                            xaxis_tickangle=-45
                        )
                        st.plotly_chart(fig_precip, use_container_width=True)
                    
                    with col2:
                        # Resumo estatístico da precipitação
                        st.markdown('<div class="precip-card">', unsafe_allow_html=True)
                        st.markdown("### 📊 Resumo de Precipitação")
                        
                        for source in precip_df['source'].unique():
                            source_data = precip_df[precip_df['source'] == source]
                            st.markdown(f"""
                            **{source}**:
                            - Total acumulado: {source_data['precip'].sum():.1f} mm
                            - Média mensal: {source_data['precip'].mean():.1f} mm
                            - Máximo mensal: {source_data['precip'].max():.1f} mm
                            """)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Comparação com área de água (apenas anos comuns)
                    st.markdown("### 🔗 Relação Precipitação x Área de Água")
                    
                    # Filtrar anos comuns
                    common_years = set(precip_years) & set(df['Ano'].tolist())
                    
                    if common_years:
                        # Agregar precipitação por ano
                        precip_df['year'] = precip_df['date'].str[:4]
                        annual_precip = precip_df.groupby(['year', 'source'])['precip'].sum().reset_index()
                        annual_precip = annual_precip[annual_precip['year'].astype(int).isin(common_years)]
                        
                        # Merge com dados de área
                        water_subset = df[df['Ano'].isin(common_years)].copy()
                        water_subset['year'] = water_subset['Ano'].astype(str)
                        
                        merged_data = pd.merge(
                            annual_precip, 
                            water_subset[['year', 'Área de Água (km²)']], 
                            on='year', 
                            how='inner'
                        )
                        
                        if not merged_data.empty:
                            fig_relation = px.scatter(
                                merged_data,
                                x='precip',
                                y='Área de Água (km²)',
                                color='source',
                                size='precip',
                                title="Relação Precipitação Anual x Área de Água",
                                labels={'precip': 'Precipitação Anual (mm)', 'Área de Água (km²)': 'Área de Água (km²)'},
                                trendline="ols"
                            )
                            fig_relation.update_layout(
                                plot_bgcolor='rgba(0,0,0,0)',
                                paper_bgcolor='rgba(0,0,0,0)',
                                height=500
                            )
                            st.plotly_chart(fig_relation, use_container_width=True)
                            
                            # Calcular correlação precipitação-água
                            for source in merged_data['source'].unique():
                                source_data = merged_data[merged_data['source'] == source]
                                if len(source_data) > 2:
                                    corr = source_data['precip'].corr(source_data['Área de Água (km²)'])
                                    st.markdown(f"""
                                    <div class="info-box">
                                        <strong>📊 Correlação ({source} vs Área de Água):</strong> r = {corr:.3f}
                                    </div>
                                    """, unsafe_allow_html=True)
                    
                else:
                    st.warning("Não foi possível carregar dados de precipitação para o período selecionado.")
                    if precip_errors:
                        st.info(f"Detalhes técnicos: {precip_errors}")
            else:
                st.info("Selecione anos na barra lateral para análise de precipitação.")

        # ===================== ANÁLISE DE CORRELAÇÃO =====================
        st.markdown("## 🔍 Análise de Correlação (Temporal)")
        
        correlation_result, error = analyze_correlation(df, correlation_method)
        
        if correlation_result:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>Coeficiente de Correlação ({correlation_method})</h3>
                    <h2>{correlation_result['coeficiente']:.3f}</h2>
                    <p>Força: {correlation_result['força']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                significance = "Significativo" if correlation_result['p_valor'] < 0.05 else "Não significativo"
                st.metric("Valor-p", f"{correlation_result['p_valor']:.4f}", significance)
            
            with col3:
                st.metric("Direção", correlation_result['direção'].title())
            
            with col4:
                st.metric("Tendência", correlation_result['tendência'].title())
            
            st.markdown(f"""
            <div class="info-box">
                <h4>📊 Interpretação da Correlação:</h4>
                <p>Existe uma correlação <strong>{correlation_result['força'].lower()}</strong> e 
                <strong>{correlation_result['direção']}</strong> entre o ano e a área de água 
                (r = {correlation_result['coeficiente']:.3f}, p = {correlation_result['p_valor']:.4f}). 
                A tendência geral é <strong>{correlation_result['tendência']}</strong> com uma taxa de 
                <strong>{abs(correlation_result['inclinação']):.3f} km²/ano</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(f"Não foi possível calcular a correlação: {error}")

        # ===================== GRÁFICOS =====================
        st.markdown("## 📈 Análise Temporal da Água")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Evolução da Área de Água")
            
            fig_line = px.line(
                df, 
                x="Ano", 
                y="Área de Água (km²)",
                title="",
                markers=True,
                color_discrete_sequence=['#2196F3'],
                line_shape='spline' if smooth_lines else 'linear'
            )
            
            if show_trendline and correlation_result:
                z = np.polyfit(df['Ano'], df['Área de Água (km²)'], 1)
                p = np.poly1d(z)
                trend_line = p(df['Ano'])
                
                fig_line.add_trace(go.Scatter(
                    x=df['Ano'],
                    y=trend_line,
                    mode='lines',
                    name=f'Tendência (r = {correlation_result["coeficiente"]:.3f})',
                    line=dict(color='#FF5252', dash='dash', width=2)
                ))
            
            fig_line.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                height=400,
                showlegend=True
            )
            
            fig_line.update_traces(line=dict(width=4), marker=dict(size=8))
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            st.markdown("### Métricas Principais")
            
            latest_year = max(selected_years)
            latest_data = df[df['Ano'] == latest_year].iloc[0]
            oldest_data = df[df['Ano'] == min(selected_years)].iloc[0]
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>📊 Área de Água ({latest_year})</h3>
                <h2>{latest_data['Área de Água (km²)']:,.1f} km²</h2>
                <p>{latest_data['Percentual de Água (%)']:.2f}% da bacia</p>
            </div>
            """, unsafe_allow_html=True)
            
            variation = ((latest_data['Área de Água (km²)'] - oldest_data['Área de Água (km²)']) / 
                       oldest_data['Área de Água (km²)']) * 100 if oldest_data['Área de Água (km²)'] > 0 else 0
            
            variation_color = "#4CAF50" if variation >= 0 else "#F44336"
            variation_icon = "📈" if variation >= 0 else "📉"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{variation_icon} Variação ({min(selected_years)}→{latest_year})</h3>
                <h2 style="color: {variation_color}">{variation:+.1f}%</h2>
                <p>Evolução no período</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de dispersão
        st.markdown("### 📊 Gráfico de Dispersão com Correlação")
        
        fig_scatter = px.scatter(
            df, 
            x="Ano", 
            y="Área de Água (km²)",
            title="",
            trendline="ols",
            trendline_color_override="red"
        )
        
        fig_scatter.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12),
            height=400
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # ===================== DOWNLOAD DE RASTERS =====================
        st.markdown("## 💾 Download de Dados")
        
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        st.markdown("### 📥 Download de Rasters")
        
        if download_mode == "Raster único":
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Configuração atual:** Ano {download_year}, Resolução: {download_resolution}m")
            with col2:
                if st.button("🗂️ Baixar Raster Único", use_container_width=True):
                    with st.spinner(f"Preparando download..."):
                        download_url, error = download_raster_bacia(download_year, download_resolution, geometry)
                        if download_url:
                            st.success(f"✅ Raster de {download_year} preparado!")
                            st.markdown(f'<a href="{download_url}" target="_blank">🎯 Clique aqui para baixar</a>', unsafe_allow_html=True)
                        else:
                            st.error(f"❌ Erro: {error}")
        
        else:
            if download_years and len(download_years) <= max_downloads:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Configuração:** {len(download_years)} anos, Resolução: {download_resolution}m")
                with col2:
                    if st.button("📦 Baixar Rasters", use_container_width=True):
                        with st.spinner("Preparando..."):
                            download_urls, error = download_multiple_rasters(
                                download_years, download_resolution, geometry, max_downloads
                            )
                            if download_urls:
                                if create_zip:
                                    zip_buffer, error = create_zip_from_urls(download_urls)
                                    if zip_buffer:
                                        st.download_button(
                                            label="📥 Baixar ZIP",
                                            data=zip_buffer.getvalue(),
                                            file_name=f"rasters_pericuma_{datetime.now().strftime('%Y%m%d')}.zip",
                                            mime="application/zip"
                                        )
                                else:
                                    for year, url in download_urls.items():
                                        st.markdown(f"**[{year}]** [📥 Baixar]({url})")
                            else:
                                st.error(f"❌ Erro: {error}")
            elif len(download_years) > max_downloads:
                st.error(f"⚠️ Selecione no máximo {max_downloads} anos.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # ===================== DOWNLOAD DE DADOS TABULARES =====================
        st.markdown("### 📊 Download de Dados Tabulares")
        
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📈 Baixar CSV com Áreas",
                data=csv,
                file_name=f"agua_pericuma_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            if correlation_result:
                report_text = f"""
RELATÓRIO DE ANÁLISE - BACIA DO RIO PERICUMÃ
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Método de correlação: {correlation_method}

RESULTADOS DA CORRELAÇÃO:
- Coeficiente: {correlation_result['coeficiente']:.3f}
- Valor-p: {correlation_result['p_valor']:.4f}
- Força: {correlation_result['força']}
- Direção: {correlation_result['direção']}
- Inclinação: {correlation_result['inclinação']:.3f} km²/ano

DADOS:
{df.to_string(index=False)}
                """
                
                st.download_button(
                    "📋 Baixar Relatório",
                    data=report_text.encode('utf-8'),
                    file_name=f"relatorio_correlacao_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        # ===================== ANÁLISE DETALHADA =====================
        st.markdown("## 📋 Análise Detalhada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dados Completos")
            styled_df = df.style.format({
                "Área de Água (km²)": "{:,.2f}",
                "Área Total da Bacia (km²)": "{:,.2f}",
                "Percentual de Água (%)": "{:.3f}%"
            }).background_gradient(subset=["Área de Água (km²)"], cmap='Blues')
            
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.markdown("### Estatísticas Descritivas")
            
            stats_summary = {
                "Média": df["Área de Água (km²)"].mean(),
                "Máximo": df["Área de Água (km²)"].max(),
                "Mínimo": df["Área de Água (km²)"].min(),
                "Desvio Padrão": df["Área de Água (km²)"].std(),
                "Coeficiente de Variação": (df["Área de Água (km²)"].std() / df["Área de Água (km²)"].mean()) * 100
            }
            
            for key, value in stats_summary.items():
                if key == "Coeficiente de Variação":
                    st.metric(key, f"{value:.1f}%")
                else:
                    st.metric(key, f"{value:,.2f} km²")

# ===================== RODAPÉ =====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🌍 <strong>Análise de Água + Precipitação - Bacia do Rio Pericumã</strong></p>
    <p>Desenvolvido com MapBiomas Água Collection 4, Google Earth Engine, GPM e CHIRPS</p>
    <p>🌧️ GPM (NASA) | 🌧️ CHIRPS (UCSB)</p>
</div>
""", unsafe_allow_html=True)