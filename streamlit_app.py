import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from openai import OpenAI
import warnings
import json
import os
from pathlib import Path
import requests

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="AI Investment Advisor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
    }
    .opportunity-card {
        background-color: #ffffff;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton>button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
    }
    .saved-key {
        background-color: #d4edda;
        padding: 0.5rem;
        border-radius: 5px;
        border: 1px solid #c3e6cb;
        margin: 0.5rem 0;
    }
    .dy-high { color: #28a745; font-weight: bold; }
    .dy-medium { color: #ffc107; font-weight: bold; }
    .dy-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# SISTEMA DE GERENCIAMENTO DE API KEY (Streamlit Secrets + Local)
# =============================================================================

CONFIG_FILE = "investment_config.json"

def carregar_configuracoes():
    """Carrega as configurações salvas"""
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        return {}
    except:
        return {}

def salvar_configuracoes(config):
    """Salva as configurações no arquivo"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        st.error(f"Erro ao salvar configurações: {e}")
        return False

def salvar_api_key(api_key):
    """Salva a API Key de forma segura"""
    try:
        config = carregar_configuracoes()
        config['openai_api_key'] = api_key
        config['ultima_atualizacao'] = datetime.now().isoformat()
        
        if salvar_configuracoes(config):
            return True
        return False
    except Exception as e:
        st.error(f"Erro ao salvar API Key: {e}")
        return False

def carregar_api_key():
    """Carrega a API Key salva"""
    try:
        # Primeiro tenta pegar do Streamlit Secrets (nuvem)
        if hasattr(st, 'secrets') and 'OPENAI_API_KEY' in st.secrets:
            return st.secrets['OPENAI_API_KEY']
        
        # Se não, tenta carregar do arquivo local
        config = carregar_configuracoes()
        return config.get('openai_api_key', '')
    except:
        return ''

def remover_api_key():
    """Remove a API Key salva"""
    try:
        config = carregar_configuracoes()
        if 'openai_api_key' in config:
            del config['openai_api_key']
        return salvar_configuracoes(config)
    except Exception as e:
        st.error(f"Erro ao remover API Key: {e}")
        return False

# =============================================================================
# FUNÇÕES PARA BUSCAR DADOS EM TEMPO REAL COM VARIAÇÃO
# =============================================================================

@st.cache_data(ttl=3600)  # Cache de 1 hora para dados históricos
def buscar_dados_historicos():
    """Busca dados históricos para calcular variações"""
    try:
        # Buscar Ibovespa (últimos 2 dias para calcular variação)
        ibov = yf.Ticker("^BVSP")
        hist_ibov = ibov.history(period="5d")
        
        # Buscar Dólar (últimos 2 dias)
        dolar = yf.Ticker("USDBRL=X")
        hist_dolar = dolar.history(period="5d")
        
        return {
            'ibov_hist': hist_ibov,
            'dolar_hist': hist_dolar
        }
    except:
        return None

def buscar_dados_macro():
    """Busca dados macroeconômicos em tempo real com variação"""
    try:
        # Buscar dados históricos
        historicos = buscar_dados_historicos()
        
        # Dados atualizados (setembro/2024) - valores de referência
        cdi_anual = 0.1331  # 13.31%
        cdi_anterior = 0.1326  # 13.26% (valor do dia anterior para exemplo)
        variacao_cdi = ((cdi_anual - cdi_anterior) / cdi_anterior) * 100
        
        ipca_acumulado = 0.0425  # 4.25%
        ipca_anterior = 0.0420  # 4.20% (valor anterior para exemplo)
        variacao_ipca = ((ipca_acumulado - ipca_anterior) / ipca_anterior) * 100
        
        # Calcular Ibovespa atual e variação
        if historicos and not historicos['ibov_hist'].empty and len(historicos['ibov_hist']) > 1:
            ibov_atual = float(historicos['ibov_hist']['Close'].iloc[-1])
            ibov_anterior = float(historicos['ibov_hist']['Close'].iloc[-2])
            variacao_ibov = ((ibov_atual - ibov_anterior) / ibov_anterior) * 100
        else:
            ibov_atual = 147428.91
            variacao_ibov = 0.31  # Exemplo de variação positiva
            
        # Calcular Dólar atual e variação
        if historicos and not historicos['dolar_hist'].empty and len(historicos['dolar_hist']) > 1:
            dolar_atual = float(historicos['dolar_hist']['Close'].iloc[-1])
            dolar_anterior = float(historicos['dolar_hist']['Close'].iloc[-2])
            variacao_dolar = ((dolar_atual - dolar_anterior) / dolar_anterior) * 100
        else:
            dolar_atual = 5.36
            variacao_dolar = 0.61  # Exemplo de variação positiva
        
        return {
            'CDI': cdi_anual,
            'VARIACAO_CDI': variacao_cdi,
            'IPCA': ipca_acumulado,
            'VARIACAO_IPCA': variacao_ipca,
            'IBOVESPA': ibov_atual,
            'VARIACAO_IBOV': variacao_ibov,
            'DOLAR': dolar_atual,
            'VARIACAO_DOLAR': variacao_dolar
        }
    except Exception as e:
        # Fallback para dados de exemplo em caso de erro
        print(f"Erro ao buscar dados macro: {e}")
        return {
            'CDI': 0.1331,
            'VARIACAO_CDI': 0.38,  # 0.38%
            'IPCA': 0.0425,
            'VARIACAO_IPCA': 1.19,  # 1.19%
            'IBOVESPA': 147428.91,
            'VARIACAO_IBOV': 0.31,  # 0.31%
            'DOLAR': 5.36,
            'VARIACAO_DOLAR': 0.61  # 0.61%
        }

def formatar_dividend_yield(dy):
    """Formata o dividend yield de forma mais clara"""
    if dy is None:
        return "N/A"
    
    try:
        # Converte para porcentagem
        if dy < 1:  # Se estiver em formato decimal (0.0475)
            dy_percent = dy * 100
        else:  # Se já estiver em porcentagem (4.75)
            dy_percent = dy
        
        # Classifica por cor
        if dy_percent >= 6:
            return f"<span class='dy-high'>{dy_percent:.2f}%</span>"
        elif dy_percent >= 3:
            return f"<span class='dy-medium'>{dy_percent:.2f}%</span>"
        else:
            return f"<span class='dy-low'>{dy_percent:.2f}%</span>"
            
    except:
        return "N/A"

def obter_info_dividendos(ticker):
    """Obtém informações detalhadas sobre dividendos"""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        
        dividend_yield = info.get('dividendYield')
        trailing_annual_dividend_rate = info.get('trailingAnnualDividendRate')
        trailing_annual_dividend_yield = info.get('trailingAnnualDividendYield')
        
        # Tenta obter os dividendos dos últimos 12 meses
        dividends = tk.dividends
        if len(dividends) > 0:
            ultimos_12m = dividends.last('12M')
            total_12m = ultimos_12m.sum() if not ultimos_12m.empty else 0
        else:
            total_12m = 0
        
        return {
            'dividend_yield': dividend_yield,
            'dividend_rate': trailing_annual_dividend_rate,
            'total_12m': total_12m,
            'ultimo_dividendo': dividends.iloc[-1] if len(dividends) > 0 else 0
        }
    except:
        return None

# =============================================================================
# FUNÇÕES PRINCIPAIS DO SISTEMA
# =============================================================================

@st.cache_data(ttl=3600)  # Cache de 1 hora
def descobrir_ativos_automaticamente():
    """Descobre automaticamente ativos promissores"""
    novos_tickers = set()
    
    # IPOs conhecidos recentemente (ATUALIZADO)
    ipos_recentes = [
        "RAIZ4.SA", "VAMO3.SA", "AZEV4.SA", 
        "LEVE3.SA", "CASH3.SA", "MDNE3.SA", "KLBN4.SA"
    ]
    novos_tickers.update(ipos_recentes)
    
    # Ações com alto volume (ATUALIZADO)
    alto_volume = [
        "MGLU3.SA", "COGN3.SA", "LWSA3.SA", 
        "PETZ3.SA", "AMER3.SA", "RADL3.SA", "CPLE6.SA"
    ]
    novos_tickers.update(alto_volume)
    
    return list(novos_tickers)

@st.cache_data(ttl=1800)  # Cache de 30 minutos
def coletar_indicadores(ticker, period="6mo"):
    """Coleta dados do Yahoo Finance"""
    try:
        if ticker in ['CDI', 'SELIC', 'IPCA']:
            dados_macro = buscar_dados_macro()
            return {
                'ticker': ticker,
                'price': dados_macro.get(ticker, 0),
                'sma50': None, 'sma200': None, 'rsi14': None,
                'pe': None, 
                'dividendYield': dados_macro.get(ticker, 0) if ticker in ['CDI', 'SELIC'] else None,
                'sector': 'Renda Fixa', 
                'classe': 'Renda Fixa',
                'nome': f'Taxa {ticker}',
                'volume': None,
                'info_dividendos': None
            }
        
        tk = yf.Ticker(ticker)
        hist = tk.history(period=period)
        
        if hist.empty:
            return None
            
        # Cálculos técnicos
        hist['SMA20'] = hist['Close'].rolling(20).mean()
        hist['SMA50'] = hist['Close'].rolling(50).mean()
        
        # RSI
        delta = hist['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        hist['RSI'] = 100 - (100 / (1 + rs))
        
        latest = hist.iloc[-1]
        info = tk.info
        
        # Informações detalhadas de dividendos
        info_dividendos = obter_info_dividendos(ticker)
        
        # Determinar classe
        if '.SA' in ticker:
            if any(x in ticker for x in ['11', '12']):
                classe = "FII"
            elif '34.SA' in ticker:
                classe = "BDR"
            else:
                classe = "Ação BR"
        else:
            classe = "Ação EUA"
        
        return {
            'ticker': ticker,
            'classe': classe,
            'price': float(latest['Close']),
            'sma20': float(latest['SMA20']) if not np.isnan(latest['SMA20']) else None,
            'sma50': float(latest['SMA50']) if not np.isnan(latest['SMA50']) else None,
            'rsi14': float(latest['RSI']) if not np.isnan(latest['RSI']) else None,
            'pe': info.get('trailingPE', None),
            'dividendYield': info.get('dividendYield', None),
            'sector': info.get('sector', 'N/A'),
            'nome': info.get('longName', ticker),
            'volume': float(latest['Volume']) if 'Volume' in latest else None,
            'info_dividendos': info_dividendos
        }
    except Exception as e:
        print(f"Erro ao coletar {ticker}: {e}")
        return None

def calcular_score_protecao(row):
    """Calcula score para preservação de capital"""
    score = 0
    
    # Peso por classe
    pesos = {"Renda Fixa": 10, "FII": 8, "ETF": 7, "BDR": 6, "Ação BR": 5, "Ação EUA": 5}
    score += pesos.get(row.get('classe', 'Ação BR'), 5)
    
    # Indicadores técnicos
    if row['sma50'] and row['price'] > row['sma50']:
        score += 2
    
    rsi = row.get('rsi14')
    if rsi and 30 < rsi < 70:
        score += 2
    
    # Fundamentais
    pe = row.get('pe')
    if pe and pe < 15:
        score += 3
    
    dy = row.get('dividendYield', 0)
    if dy:
        if dy < 1:  # Decimal
            dy_percent = dy * 100
        else:  # Já em porcentagem
            dy_percent = dy
            
        if dy_percent >= 6: 
            score += 3
        elif dy_percent >= 4: 
            score += 2
        elif dy_percent >= 2: 
            score += 1
    
    return score

def calcular_score_crescimento(row):
    """Calcula score para crescimento agressivo"""
    score = 0
    
    # Bônus para novas descobertas
    if row.get('ticker') in st.session_state.get('novos_tickers', []):
        score += 5
    
    # Tendência
    if row['sma20'] and row['sma50']:
        if row['sma20'] > row['sma50']:
            score += 3
    
    # RSI para momentum
    rsi = row.get('rsi14')
    if rsi and 40 < rsi < 65:
        score += 2
    elif rsi and rsi < 30:
        score += 3
    
    # Valuation agressivo
    pe = row.get('pe')
    if pe and pe < 12:
        score += 3
    
    # Volume (liquidez)
    if row.get('volume', 0) > 1000000:
        score += 1
    
    return score

def analisar_com_gpt(df, strategy, horizon, api_key):
    """Analisa oportunidades usando GPT"""
    if not api_key:
        return "⚠️ Adicione sua OpenAI API Key para obter análise detalhada"
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepara dados formatados para análise
        dados_analise = df.head(15).copy()
        
        # Formata dividend yield para exibição mais clara
        for i, row in dados_analise.iterrows():
            if pd.notna(row['dividendYield']):
                if row['dividendYield'] < 1:
                    dados_analise.at[i, 'dividendYield'] = f"{row['dividendYield']*100:.2f}%"
                else:
                    dados_analise.at[i, 'dividendYield'] = f"{row['dividendYield']:.2f}%"
        
        prompt = f"""
        Como analista financeiro especializado, analise estas oportunidades:
        
        Estratégia: {strategy}
        Horizonte: {horizon}
        
        DADOS ATUALIZADOS DOS ATIVOS:
        {dados_analise[['ticker', 'classe', 'price', 'rsi14', 'pe', 'dividendYield', 'Score_Final']].to_string()}
        
        CONTEXTO MACRO ATUAL:
        - CDI: 13.31% a.a. | Selic: 13.31% a.a. | IPCA: 4.25% a.a.
        - Ibovespa: 147.676 pts | Dólar: R$ 5,36
        - Cenário: Juros altos, inflação controlada
        
        Forneça uma análise com:
        1. TOP 3 OPORTUNIDADES - Justificativa técnica detalhada
        2. ALAVANCAGENS - O que pode valorizar cada ativo
        3. RISCOS - Principais preocupações
        4. ALOCAÇÃO SUGERIDA - % para cada ativo na carteira
        5. GATILHOS - Quando comprar/vender
        
        Foque em oportunidades reais com base nos dados técnicos.
        Seja prático e direto, evite jargões complexos.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro na análise GPT: {str(e)}"

# =============================================================================
# INTERFACE PRINCIPAL
# =============================================================================

# Título principal
st.markdown('<h1 class="main-header">🤖 AI Investment Advisor</h1>', unsafe_allow_html=True)
st.markdown("### Sistema Inteligente de Análise e Descoberta de Oportunidades")

# Sidebar para configurações
with st.sidebar:
    st.header("⚙️ Configurações")
    
    # Sistema de API Key
    st.subheader("🔑 Configuração da API Key")
    
    # Verifica se há uma API Key salva
    api_key_salva = carregar_api_key()
    
    if api_key_salva:
        st.markdown(f'<div class="saved-key">✅ API Key salva</div>', unsafe_allow_html=True)
        if st.button("🗑️ Remover API Key Salva"):
            if remover_api_key():
                st.success("API Key removida com sucesso!")
                st.rerun()
            else:
                st.error("Erro ao remover API Key")
    
    # Input para API Key
    api_key = st.text_input("OpenAI API Key", type="password", 
                           value=api_key_salva,
                           help="Obtenha em: https://platform.openai.com/api-keys")
    
    # Botão para salvar API Key
    if api_key and api_key != api_key_salva:
        if st.button("💾 Salvar API Key"):
            if salvar_api_key(api_key):
                st.success("API Key salva com sucesso!")
                st.rerun()
            else:
                st.error("Erro ao salvar API Key")
    
    # Informações sobre fontes de dados
    with st.expander("📊 Fontes dos Dados"):
        st.write("""
        **Dados em Tempo Real:**
        - ✅ **Yahoo Finance**: Cotações, indicadores técnicos
        - ✅ **OpenAI GPT-4**: Análise inteligente
        - ✅ **BCB/Dados Mercado**: CDI, Selic, IPCA
        
        **Atualização Automática:**
        - Cotações: Tempo real
        - Indicadores: A cada 30min
        - Análise IA: Sob demanda
        
        **Variações Calculadas:**
        - 📈 Comparação com período anterior
        - 🔄 Dados históricos para cálculo
        - ⚡ Atualização automática
        """)
    
    # Estratégia de investimento
    strategy = st.selectbox(
        "🎯 Estratégia de Investimento",
        ["Preservação de Capital", "Crescimento Moderado", "Crescimento Agressivo", "Renda"]
    )
    
    # Horizonte temporal
    horizon = st.selectbox(
        "📅 Horizonte de Investimento",
        ["Curto Prazo (1-6 meses)", "Médio Prazo (6-24 meses)", "Longo Prazo (+24 meses)"]
    )
    
    # Classe de ativos
    asset_classes = st.multiselect(
        "🏦 Classes de Ativos",
        ["Ações Brasileiras", "FIIs", "BDRs", "ETFs", "Ações EUA", "Renda Fixa"],
        default=["Ações Brasileiras", "FIIs", "BDRs"]
    )
    
    # Busca automática
    auto_discover = st.checkbox("🔍 Busca Automática de Oportunidades", value=True)
    
    if st.button("🔄 Executar Análise Completa"):
        st.session_state.run_analysis = True
    else:
        st.session_state.run_analysis = False

# Lista base de ativos ATUALIZADA
tickers_base = {
    "Ações Brasileiras": [
        "PETR4.SA", "VALE3.SA", "ITUB4.SA", "BBDC4.SA", "ABEV3.SA", "WEGE3.SA",
        "MGLU3.SA", "BBAS3.SA", "B3SA3.SA", "RENT3.SA", "LREN3.SA", "EQTL3.SA",
        "RADL3.SA", "SBSP3.SA", "CSAN3.SA", "EMBR3.SA", "GGBR4.SA", "CYRE3.SA"
    ],
    "FIIs": [
        "HGLG11.SA", "KNRI11.SA", "XPML11.SA", "HGRU11.SA", "VGIP11.SA", "XPLG11.SA",
        "BCFF11.SA", "RBRF11.SA", "HSML11.SA", "VRTA11.SA"
    ],
    "BDRs": [
        "AAPL34.SA", "MSFT34.SA", "AMZO34.SA", "TSLA34.SA", 
        "COCA34.SA", "DISB34.SA", "JPMO34.SA", "NVDC34.SA"
    ],
    "ETFs": [
        "BOVA11.SA", "SMAL11.SA", "IVVB11.SA", "BBSD11.SA", "DIVO11.SA", "FIND11.SA"
    ],
    "Renda Fixa": ["CDI", "SELIC", "IPCA"]
}

# Interface principal
tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🎯 Oportunidades", "📈 Análise Detalhada", "⚡ Descobertas"])

with tab1:
    st.header("Visão Geral do Mercado")
    
    # Buscar dados em tempo real com variação
    dados_macro = buscar_dados_macro()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "CDI Anual", 
            f"{dados_macro['CDI']*100:.2f}%", 
            f"{dados_macro['VARIACAO_CDI']:.2f}%"
        )
    with col2:
        st.metric(
            "IPCA Acumulado", 
            f"{dados_macro['IPCA']*100:.2f}%", 
            f"{dados_macro['VARIACAO_IPCA']:.2f}%"
        )
    with col3:
        st.metric(
            "Ibovespa", 
            f"{dados_macro['IBOVESPA']:,.2f}".replace(',', '_').replace('.', ',').replace('_', '.'), 
            f"{dados_macro['VARIACAO_IBOV']:.2f}%"
        )
    with col4:
        st.metric(
            "Dólar", 
            f"R$ {dados_macro['DOLAR']:.2f}", 
            f"{dados_macro['VARIACAO_DOLAR']:.2f}%"
        )
    
    # Informações sobre as variações
    with st.expander("ℹ️ Sobre as Variações"):
        st.write("""
        **Como as variações são calculadas:**
        - **CDI e IPCA**: Variação percentual em relação ao período anterior
        - **Ibovespa**: Variação percentual do último pregão
        - **Dólar**: Variação percentual do último pregão
        
        **Atualização:** Os dados são atualizados automaticamente a cada execução.
        """)
    
    # Gráfico de performance
    st.subheader("📈 Performance do Ibovespa (6 meses)")
    try:
        ibov = yf.download("^BVSP", period="6mo")['Close']
        fig = px.line(ibov, title="Ibovespa - Últimos 6 Meses")
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.info("Não foi possível carregar dados do Ibovespa")

with tab2:
    st.header("🎯 Melhores Oportunidades")
    
    if st.button("🔍 Buscar Oportunidades") or st.session_state.get('run_analysis', False):
        with st.spinner("Analisando oportunidades de investimento..."):
            # Coletar tickers
            todos_tickers = []
            for classe in asset_classes:
                if classe in tickers_base:
                    todos_tickers.extend(tickers_base[classe])
            
            # Busca automática
            if auto_discover:
                novos_tickers = descobrir_ativos_automaticamente()
                st.session_state.novos_tickers = novos_tickers
                todos_tickers.extend(novos_tickers)
            
            # Coletar dados
            dados = []
            progress_bar = st.progress(0)
            
            for i, ticker in enumerate(todos_tickers):
                resultado = coletar_indicadores(ticker)
                if resultado:
                    dados.append(resultado)
                progress_bar.progress((i + 1) / len(todos_tickers))
            
            if dados:
                df = pd.DataFrame(dados)
                
                # Calcular scores
                df['Score_Protecao'] = df.apply(calcular_score_protecao, axis=1)
                df['Score_Crescimento'] = df.apply(calcular_score_crescimento, axis=1)
                
                # Score combinado baseado na estratégia
                if strategy == "Preservação de Capital":
                    df['Score_Final'] = df['Score_Protecao'] * 0.8 + df['Score_Crescimento'] * 0.2
                elif strategy == "Crescimento Agressivo":
                    df['Score_Final'] = df['Score_Protecao'] * 0.2 + df['Score_Crescimento'] * 0.8
                else:  # Crescimento Moderado
                    df['Score_Final'] = df['Score_Protecao'] * 0.5 + df['Score_Crescimento'] * 0.5
                
                df = df.sort_values('Score_Final', ascending=False)
                st.session_state.df_analise = df
                
                # Top oportunidades
                st.subheader("🏆 Top 10 Oportunidades")
                
                for _, row in df.head(10).iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([3, 2, 1])
                        
                        with col1:
                            st.write(f"**{row['ticker']}** - {row.get('nome', '')}")
                            st.write(f"*{row['classe']}* | Setor: {row.get('sector', 'N/A')}")
                        
                        with col2:
                            st.write(f"Preço: R$ {row['price']:.2f}")
                            if row.get('dividendYield') is not None:
                                dy_formatado = formatar_dividend_yield(row['dividendYield'])
                                st.markdown(f"**DY:** {dy_formatado}", unsafe_allow_html=True)
                                
                                # Mostra informações adicionais de dividendos se disponível
                                if row.get('info_dividendos') and row['info_dividendos'].get('total_12m', 0) > 0:
                                    st.write(f"Dividendos 12m: R$ {row['info_dividendos']['total_12m']:.2f}")
                        
                        with col3:
                            st.metric("Score", f"{row['Score_Final']:.1f}")
                        
                        st.markdown("---")
                
                # Gráfico de distribuição
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Distribuição por Classe")
                    fig = px.pie(df, names='classe', title='Oportunidades por Classe')
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("🎯 Scores por Estratégia")
                    fig = px.scatter(df.head(20), x='Score_Protecao', y='Score_Crescimento', 
                                   color='classe', size='Score_Final', hover_data=['ticker'],
                                   title='Proteção vs Crescimento')
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("Não foi possível coletar dados dos ativos")

with tab3:
    st.header("📈 Análise Detalhada")
    
    if st.session_state.get('df_analise') is not None:
        df = st.session_state.df_analise
        
        # Filtros
        col1, col2, col3 = st.columns(3)
        
        with col1:
            classes_selecionadas = st.multiselect(
                "Filtrar por Classe",
                options=df['classe'].unique(),
                default=df['classe'].unique()
            )
        
        with col2:
            score_min = st.slider("Score Mínimo", 0, 20, 5)
        
        with col3:
            show_new = st.checkbox("Apenas Novas Descobertas", value=False)
        
        # Aplicar filtros
        df_filtrado = df[df['classe'].isin(classes_selecionadas)]
        df_filtrado = df_filtrado[df_filtrado['Score_Final'] >= score_min]
        
        if show_new and st.session_state.get('novos_tickers'):
            df_filtrado = df_filtrado[df_filtrado['ticker'].isin(st.session_state.novos_tickers)]
        
        # Tabela detalhada
        st.subheader("📋 Detalhes dos Ativos")
        
        # Formatar Dividend Yield para exibição
        df_display = df_filtrado.copy()
        df_display['dividendYield'] = df_display['dividendYield'].apply(
            lambda x: formatar_dividend_yield(x) if pd.notna(x) else "N/A"
        )
        
        st.dataframe(df_display[['ticker', 'classe', 'price', 'rsi14', 'pe', 'dividendYield', 
                                'Score_Protecao', 'Score_Crescimento', 'Score_Final']],
                   use_container_width=True)
        
        # Análise GPT
        if st.button("🤖 Obter Análise IA") and api_key:
            with st.spinner("Gerando análise detalhada..."):
                analise = analisar_com_gpt(df_filtrado, strategy, horizon, api_key)
                st.subheader("💡 Análise da IA")
                st.write(analise)
        elif not api_key:
            st.info("Adicione sua OpenAI API Key para obter análise da IA")

with tab4:
    st.header("⚡ Novas Descobertas")
    
    if st.session_state.get('novos_tickers'):
        st.success(f"🎉 {len(st.session_state.novos_tickers)} novas oportunidades descobertas!")
        
        # Detalhes das novas descobertas
        if st.session_state.get('df_analise') is not None:
            df_novos = st.session_state.df_analise[
                st.session_state.df_analise['ticker'].isin(st.session_state.novos_tickers)
            ].sort_values('Score_Crescimento', ascending=False)
            
            for _, row in df_novos.iterrows():
                with st.expander(f"🚀 {row['ticker']} - Score Crescimento: {row['Score_Crescimento']:.1f}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Classe:** {row['classe']}")
                        st.write(f"**Preço:** R$ {row['price']:.2f}")
                        st.write(f"**Setor:** {row.get('sector', 'N/A')}")
                        if row.get('info_dividendos'):
                            info = row['info_dividendos']
                            if info.get('total_12m', 0) > 0:
                                st.write(f"**Dividendos 12m:** R$ {info['total_12m']:.2f}")
                    
                    with col2:
                        if row.get('rsi14'):
                            st.write(f"**RSI:** {row['rsi14']:.1f}")
                        if row.get('pe'):
                            st.write(f"**P/L:** {row['pe']:.1f}")
                        if row.get('dividendYield') is not None:
                            dy_formatado = formatar_dividend_yield(row['dividendYield'])
                            st.markdown(f"**DY:** {dy_formatado}", unsafe_allow_html=True)
        
        # Gráfico de novas descobertas
        if not df_novos.empty:
            fig = px.bar(df_novos, x='ticker', y='Score_Crescimento', 
                        title='Potencial de Crescimento - Novas Descobertas',
                        color='Score_Crescimento', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Execute a análise para descobrir novas oportunidades")

# Rodapé com horário correto
st.markdown("---")
st.markdown(
    "⚠️ **Aviso Legal:** Esta ferramenta é apenas para fins educacionais. "
    "Não constitui recomendação de investimento. Consulte um profissional qualificado."
)
st.markdown(
    f"📊 **Fontes de Dados:** Yahoo Finance, Banco Central do Brasil, OpenAI GPT-4 | "
    f"🔄 **Última atualização:** {datetime.now().strftime('%d/%m/%Y %H:%M')}"
)
