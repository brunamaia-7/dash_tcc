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

# Inicialização do Earth Engine
# ===================== CONFIGURAÇÃO EARTH ENGINE =====================
def initialize_earth_engine():
    try:
        ee.Initialize(project='ee-brunamaiiia')  # 🔥 FORÇA SEU PROJETO
        st.sidebar.success("✅ Earth Engine inicializado (projeto correto)")
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
                
                ee.Initialize(credentials, project='ee-brunamaiiia')  # 🔥 AQUI TAMBÉM
                st.sidebar.success("✅ Earth Engine inicializado (Service Account)")
                return True
            else:
                st.sidebar.warning("⚠️ Earth Engine sem autenticação")
                return False
        except Exception as e:
            st.sidebar.error(f"❌ Erro real: {str(e)}")
            return False

# Inicializar Earth Engine
ee_initialized = initialize_earth_engine()

# Configuração da página
st.set_page_config(
    layout='wide',
    page_title="🌊 Análise de Água - Bacia do Pericumã",
    page_icon="🌊"
)

# ===================== ESTILOS E CSS =====================
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
    .correlation-high {
        background-color: #E8F5E8 !important;
    }
    .correlation-medium {
        background-color: #FFF8E1 !important;
    }
    .correlation-low {
        background-color: #FFEBEE !important;
    }
    .download-section {
        background: #F5F5F5;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===================== CABEÇALHO =====================
st.markdown('<h1 class="main-header">WebApp para monitoramento da superfície de água na bacia hidrográfica do rio Pericumã</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Monitoramento da dinâmica de corpos d\'água através do MapBiomas Água Collection 4</p>', unsafe_allow_html=True)

# ===================== CONFIGURAÇÃO DA BACIA =====================
PERICUMA_ASSET = 'projects/ee-brunamaiiia/assets/Bacia_Pericuma_ZEE_v2'
bacia = ee.FeatureCollection(PERICUMA_ASSET)
geometry = bacia.geometry()

# ===================== CARREGAR DADOS MAPBIOMAS ÁGUA =====================
water_image = ee.Image('projects/mapbiomas-public/assets/brazil/water/collection4/mapbiomas_brazil_collection4_water_v3')
WATER_VALUE = 1

# ===================== SIDEBAR =====================
with st.sidebar:
    # Logos no topo do sidebar (pequenas, lado a lado)
    col1, col2 = st.columns([1, 1])
    with col1:
        st.image("assets/lageos.jpeg", width=80)
    with col2:
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
        ["Raster único", "Múltiplos rasters"],
        help="Escolha entre baixar um único raster ou vários de uma vez"
    )
    
    if download_mode == "Raster único":
        download_year = st.selectbox(
            "Ano para download do raster",
            options=years,
            index=len(years)-1,
            help="Selecione o ano para baixar o raster da bacia"
        )
    else:
        download_years = st.multiselect(
            "Selecione os anos para download",
            options=years,
            default=[2022, 2020, 2018],
            help="Selecione múltiplos anos para download em lote"
        )
    
    download_resolution = st.slider(
        "Resolução do raster (metros)",
        min_value=30,
        max_value=500,
        value=100,
        step=10,
        help="Resolução espacial para o download"
    )
    
    # Opções adicionais para download múltiplo
    if download_mode == "Múltiplos rasters":
        create_zip = st.checkbox(
            "📦 Compactar em arquivo ZIP", 
            value=True,
            help="Cria um arquivo ZIP com todos os rasters selecionados"
        )
        
        max_downloads = st.slider(
            "Número máximo de rasters por download",
            min_value=1,
            max_value=10,
            value=5,
            help="Limite para evitar downloads muito grandes"
        )
    
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
    - Resolução espacial: 30 metros
    - Período: 1985–2024
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
        
        # Recortar para a bacia
        raster_clipped = water_band.clip(geometry)
        
        # Configurar parâmetros de exportação
        export_params = {
            'scale': resolution,
            'region': geometry,
            'fileFormat': 'GeoTIFF',
            'formatOptions': {
                'cloudOptimized': True
            }
        }
        
        # Criar URL de download
        download_url = raster_clipped.getDownloadUrl(export_params)
        
        return download_url, None
        
    except Exception as e:
        return None, str(e)

def download_multiple_rasters(years, resolution, geometry, max_downloads=5):
    """Baixa múltiplos rasters de uma vez e retorna URLs"""
    try:
        if len(years) > max_downloads:
            return None, f"Selecione no máximo {max_downloads} anos por download"
        
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
    """Cria um arquivo ZIP a partir dos URLs de download"""
    try:
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for year, url in download_urls.items():
                # Fazer download do arquivo
                response = requests.get(url)
                if response.status_code == 200:
                    filename = f"agua_pericuma_{year}.tif"
                    zip_file.writestr(filename, response.content)
                else:
                    return None, f"Erro ao baixar arquivo do ano {year}"
        
        zip_buffer.seek(0)
        return zip_buffer, None
        
    except Exception as e:
        return None, str(e)

def get_individual_download_links(download_urls):
    """Gera links individuais para download"""
    links_html = ""
    for year, url in download_urls.items():
        links_html += f"""
        <div style="margin: 5px 0;">
            <strong>Ano {year}:</strong> 
            <a href="{url}" target="_blank" style="margin-left: 10px;">📥 Baixar raster {year}</a>
        </div>
        """
    return links_html

# ===================== FUNÇÃO DE ANÁLISE DE CORRELAÇÃO =====================
def analyze_correlation(df, method='Pearson'):
    """Realiza análise de correlação entre ano e área de água"""
    try:
        if len(df) < 2:
            return None, "Dados insuficientes para análise de correlação"
        
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
        
        # Classificar a correlação
        abs_corr = abs(corr_coef)
        if abs_corr >= 0.7:
            strength = "Forte"
            strength_class = "correlation-high"
        elif abs_corr >= 0.3:
            strength = "Moderada"
            strength_class = "correlation-medium"
        else:
            strength = "Fraca"
            strength_class = "correlation-low"
        
        # Determinar direção
        direction = "positiva" if corr_coef > 0 else "negativa"
        
        # Calcular regressão linear para tendência
        slope, intercept = np.polyfit(x, y, 1)
        trend = "crescente" if slope > 0 else "decrescente"
        
        correlation_result = {
            'coeficiente': corr_coef,
            'p_valor': p_value,
            'força': strength,
            'direção': direction,
            'tendência': trend,
            'classe_css': strength_class,
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

    m.add_basemap('OpenStreetMap')

    # Adicionar camadas de água para anos selecionados
    if selected_years:
        for year in selected_years:
            area_km2, water_mask = calculate_water_area(year)
            if water_mask:
                water_clipped = water_mask.clip(geometry)
                m.addLayer(
                    water_clipped,
                    {
                        'palette': ['00000000', '#1976D2'],
                        'min': 0,
                        'max': 1
                    },
                    f"Água {year} ({area_km2:.1f} km²)"
                )

    # Adicionar mapas base
    m.add_basemap('OpenStreetMap')
    m.add_basemap('Google Terrain')
    m.add_basemap('Google Satellite')
    
    # Adicionar controles do mapa
    m.add_layer_control()

    # Container estilizado para o mapa
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
        # ===================== ANÁLISE DE CORRELAÇÃO =====================
        st.markdown("## 🔍 Análise de Correlação")
        
        correlation_result, error = analyze_correlation(df, correlation_method)
        
        if correlation_result:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="metric-card {correlation_result['classe_css']}">
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
            
            # Interpretação da correlação
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
        
        # Container principal para gráficos
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("### Evolução da Área de Água")
            
            # Gráfico de linha suave com tendência
            fig_line = px.line(
                df, 
                x="Ano", 
                y="Área de Água (km²)",
                title="",
                markers=True,
                color_discrete_sequence=['#2196F3'],
                line_shape='spline' if smooth_lines else 'linear'
            )
            
            if show_trendline:
                # Adicionar linha de tendência
                z = np.polyfit(df['Ano'], df['Área de Água (km²)'], 1)
                p = np.poly1d(z)
                trend_line = p(df['Ano'])
                
                fig_line.add_trace(go.Scatter(
                    x=df['Ano'],
                    y=trend_line,
                    mode='lines',
                    name=f'Tendência (r = {correlation_result["coeficiente"]:.3f})' if correlation_result else 'Tendência',
                    line=dict(color='#FF5252', dash='dash', width=2),
                    hoverinfo='skip'
                ))
            
            fig_line.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12),
                height=400,
                showlegend=True
            )
            
            fig_line.update_traces(
                line=dict(width=4),
                marker=dict(size=8)
            )
            
            st.plotly_chart(fig_line, use_container_width=True)
        
        with col2:
            st.markdown("### Métricas Principais")
            
            latest_year = max(selected_years)
            latest_data = df[df['Ano'] == latest_year].iloc[0]
            oldest_data = df[df['Ano'] == min(selected_years)].iloc[0]
            
            # Cartões de métricas
            st.markdown(f"""
            <div class="metric-card">
                <h3> Área de Água ({latest_year})</h3>
                <h2>{latest_data['Área de Água (km²)']:,.1f} km²</h2>
                <p>{latest_data['Percentual de Água (%)']:.2f}% da bacia</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Variação
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
            
            # Métrica de correlação se disponível
            if correlation_result:
                st.markdown(f"""
                <div class="metric-card {correlation_result['classe_css']}">
                    <h3>📊 Correlação</h3>
                    <h2>{correlation_result['coeficiente']:.3f}</h2>
                    <p>{correlation_result['força']} • {correlation_result['direção']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Gráfico de dispersão com linha de correlação
        st.markdown("### Gráfico de Dispersão com Correlação")
        
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
                    with st.spinner(f"Preparando download do raster {download_year}..."):
                        download_url, error = download_raster_bacia(download_year, download_resolution, geometry)
                        
                        if download_url:
                            st.success(f"✅ Raster de {download_year} preparado para download!")
                            st.markdown(f"""
                            <div class="success-box">
                                <h4>📋 Detalhes do Download:</h4>
                                <p><strong>Ano:</strong> {download_year}</p>
                                <p><strong>Resolução:</strong> {download_resolution} metros</p>
                                <p><strong>Formato:</strong> GeoTIFF</p>
                                <p><strong>Tamanho estimado:</strong> ~10-50 MB</p>
                                <p><a href="{download_url}" target="_blank" style="color: #2196F3; font-weight: bold;">🎯 Clique aqui para baixar o arquivo</a></p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.error(f"❌ Erro ao preparar download: {error}")
        
        else:  # Múltiplos rasters
            if not download_years:
                st.warning("⚠️ Selecione pelo menos um ano para download múltiplo.")
            else:
                if len(download_years) > max_downloads:
                    st.error(f"⚠️ Selecione no máximo {max_downloads} anos por download.")
                else:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**Configuração atual:** {len(download_years)} anos selecionados, Resolução: {download_resolution}m")
                        st.markdown(f"**Anos:** {', '.join(map(str, sorted(download_years)))}")
                    
                    with col2:
                        download_button_text = "📦 Baixar ZIP" if create_zip else "🗂️ Baixar Rasters"
                        if st.button(download_button_text, use_container_width=True):
                            with st.spinner(f"Preparando download de {len(download_years)} rasters..."):
                                download_urls, error = download_multiple_rasters(
                                    download_years, download_resolution, geometry, max_downloads
                                )
                                
                                if download_urls:
                                    if create_zip:
                                        # Criar arquivo ZIP
                                        zip_buffer, error = create_zip_from_urls(download_urls)
                                        if zip_buffer:
                                            st.success(f"✅ ZIP com {len(download_urls)} rasters preparado!")
                                            
                                            # Botão para download do ZIP
                                            zip_filename = f"rasters_pericuma_{datetime.now().strftime('%Y%m%d_%H%M')}.zip"
                                            st.download_button(
                                                label="📥 Baixar Arquivo ZIP",
                                                data=zip_buffer.getvalue(),
                                                file_name=zip_filename,
                                                mime="application/zip",
                                                use_container_width=True
                                            )
                                            
                                            st.markdown(f"""
                                            <div class="success-box">
                                                <h4>📦 Conteúdo do ZIP:</h4>
                                                <p><strong>Arquivos incluídos:</strong> {len(download_urls)} rasters</p>
                                                <p><strong>Anos:</strong> {', '.join(map(str, sorted(download_years)))}</p>
                                                <p><strong>Resolução:</strong> {download_resolution}m</p>
                                                <p><strong>Tamanho estimado:</strong> ~{len(download_urls)*20} MB</p>
                                            </div>
                                            """, unsafe_allow_html=True)
                                        else:
                                            st.error(f"❌ Erro ao criar ZIP: {error}")
                                    else:
                                        # Links individuais
                                        st.success(f"✅ {len(download_urls)} rasters preparados para download!")
                                        st.markdown(f"""
                                        <div class="success-box">
                                            <h4>🔗 Links de Download Individuais:</h4>
                                            {get_individual_download_links(download_urls)}
                                        </div>
                                        """, unsafe_allow_html=True)
                                else:
                                    st.error(f"❌ Erro ao preparar downloads: {error}")
        
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
            # Download do relatório de correlação
            if correlation_result:
                report_text = f"""
RELATÓRIO DE ANÁLISE - BACIA DO RIO PERICUMÃ
Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}
Método de correlação: {correlation_method}

RESULTADOS DA CORRELAÇÃO:
- Coeficiente de correlação: {correlation_result['coeficiente']:.3f}
- Valor-p: {correlation_result['p_valor']:.4f}
- Força: {correlation_result['força']}
- Direção: {correlation_result['direção']}
- Tendência: {correlation_result['tendência']}
- Inclinação: {correlation_result['inclinação']:.3f} km²/ano

DADOS BRUTOS:
{df.to_string(index=False)}
                """
                
                st.download_button(
                    "📋 Baixar Relatório de Correlação",
                    data=report_text.encode('utf-8'),
                    file_name=f"relatorio_correlacao_pericuma_{datetime.now().strftime('%Y%m%d')}.txt",
                    mime="text/plain",
                    use_container_width=True
                )

        # ===================== ANÁLISE DETALHADA =====================
        st.markdown("## 📋 Análise Detalhada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Dados Completos")
            
            # Aplicar estilo baseado na correlação
            def style_correlation(val):
                if correlation_result:
                    if abs(correlation_result['coeficiente']) >= 0.7:
                        return 'background-color: #E8F5E8'
                    elif abs(correlation_result['coeficiente']) >= 0.3:
                        return 'background-color: #FFF8E1'
                    else:
                        return 'background-color: #FFEBEE'
                return ''
            
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
                "Coeficiente de Variação": (df["Área de Água (km²)"].std() / df["Área de Água (km²)"].mean()) * 100,
                "Variação Total": f"{variation:+.1f}%"
            }
            
            for key, value in stats_summary.items():
                if isinstance(value, float):
                    if key == "Coeficiente de Variação":
                        st.metric(key, f"{value:.1f}%")
                    else:
                        st.metric(key, f"{value:,.2f} km²")
                else:
                    st.metric(key, value)

# ===================== RODAPÉ =====================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <p>🌍 <strong>Análise de Água - Bacia do Rio Pericumã</strong></p>
    <p>Desenvolvido com MapBiomas Água Collection 4 e Google Earth Engine</p>
    <p>📅 Dados atualizados: 1985-2023 | 📏 Resolução: 30 metros</p>
    <p>🔍 Análise de correlação: Pearson, Spearman e Kendall</p>
    <p>💾 Download: Rasters individuais ou múltiplos em lote</p>
</div>
""", unsafe_allow_html=True)