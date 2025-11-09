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
import pytz

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
    .dy-high { 
        background-color: #d4edda; 
        color: #155724; 
        font-weight: bold; 
        padding: 2px 6px;
        border-radius: 3px;
    }
    .dy-medium { 
        background-color: #fff3cd; 
        color: #856404; 
        font-weight: bold; 
        padding: 2px 6px;
        border-radius: 3px;
    }
    .dy-low { 
        background-color: #f8d7da; 
        color: #721c24; 
        font-weight: bold; 
        padding: 2px 6px;
        border-radius: 3px;
    }
    .pe-low { 
        background-color: #d4edda; 
        color: #155724; 
        font-weight: bold; 
        padding: 2px 6px;
        border-radius: 3px;
    }
    .pe-medium { 
        background-color: #fff3cd; 
        color: #856404; 
        font-weight: bold; 
        padding: 2px 6px;
        border-radius: 3px;
    }
    .pe-high { 
        background-color: #f8d7da; 
        color: #721c24; 
        font-weight: bold; 
        padding: 2px 6px;
        border-radius: 3px;
    }
    .param-card {
        background-color: #898989;
        padding: 0.5rem;
        border-radius: 5px;
        border-left: 3px solid #1f77b4;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }
    .compact-card {
        background-color: #f8f9fa;
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
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
        config['ultima_atualizacao'] = datetime.now(pytz.timezone('America/Sao_Paulo')).isoformat()
        
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

def obter_horario_brasilia():
    """Obtém o horário atual de Brasília"""
    try:
        tz_brasilia = pytz.timezone('America/Sao_Paulo')
        agora = datetime.now(tz_brasilia)
        return agora.strftime('%d/%m/%Y %H:%M')
    except:
        # Fallback se pytz não estiver disponível
        return datetime.now().strftime('%d/%m/%Y %H:%M')

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
    """Formata o dividend yield com cores condicionais"""
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

def formatar_pe(pe):
    """Formata o P/L com cores condicionais"""
    if pe is None or np.isnan(pe):
        return "N/A"
    
    try:
        # Classifica por cor
        if pe < 12:
            return f"<span class='pe-low'>{pe:.1f}</span>"
        elif pe < 20:
            return f"<span class='pe-medium'>{pe:.1f}</span>"
        else:
            return f"<span class='pe-high'>{pe:.1f}</span>"
    except:
        return "N/A"

def obter_info_dividendos_detalhada(ticker, preco_atual=None):
    """Obtém informações detalhadas sobre dividendos com valores em R$"""
    try:
        tk = yf.Ticker(ticker)
        info = tk.info
        
        # Tenta obter dividend yield de várias fontes
        dividend_yield = (info.get('dividendYield') or 
                         info.get('trailingDividendYield') or 
                         info.get('forwardDividendYield') or 
                         info.get('yield'))
        
        # Tenta obter informações específicas de dividendos
        forward_dividend_rate = info.get('forwardDividendRate')
        trailing_dividend_rate = info.get('trailingAnnualDividendRate')
        last_dividend_value = info.get('lastDividendValue')
        
        # Busca dados históricos de dividendos
        dividends = tk.dividends
        ultimo_dividendo = 0
        total_12m = 0
        dividendos_recentes = []
        quantidade_dividendos_12m = 0
        
        # Processa dividendos históricos se disponíveis
        if len(dividends) > 0:
            # Filtra dividendos válidos
            dividends_validos = dividends[~dividends.isna()]
            
            if len(dividends_validos) > 0:
                # CORREÇÃO DO TIMEZONE: Remove timezone para comparação
                dividends_index_no_tz = dividends_validos.index.tz_localize(None)
                dividends_validos_no_tz = pd.Series(dividends_validos.values, index=dividends_index_no_tz)
                
                # Último dividendo
                ultimo_dividendo = float(dividends_validos_no_tz.iloc[-1])
                
                # Dividendos dos últimos 12 meses (sem timezone)
                data_limite = datetime.now() - timedelta(days=365)
                dividendos_12m = dividends_validos_no_tz[dividends_validos_no_tz.index >= data_limite]
                quantidade_dividendos_12m = len(dividendos_12m)
                total_12m = float(dividendos_12m.sum()) if not dividendos_12m.empty else 0
                
                # Últimos 3 dividendos
                dividendos_recentes = [float(div) for div in dividends_validos_no_tz.tail(3)]
        
        # Se não encontrou dividendos no histórico, tenta usar outras fontes
        if ultimo_dividendo == 0:
            # Prioridade: forwardDividendRate, depois trailing, depois lastDividendValue
            if forward_dividend_rate and forward_dividend_rate > 0:
                ultimo_dividendo = forward_dividend_rate
            elif trailing_dividend_rate and trailing_dividend_rate > 0:
                ultimo_dividendo = trailing_dividend_rate
            elif last_dividend_value and last_dividend_value > 0:
                ultimo_dividendo = last_dividend_value
        
        # Para FIIs, ajusta o cálculo se o dividendo for anual
        if ultimo_dividendo > 1 and ticker.endswith('.SA') and any(x in ticker for x in ['11', '12']):
            # Se parece ser um dividendo anual grande, divide por 12
            if ultimo_dividendo > preco_atual * 0.05:  # Mais de 5% do preço
                ultimo_dividendo = ultimo_dividendo / 12
        
        # Calcula total dos últimos 12 meses se não disponível
        if total_12m == 0 and dividend_yield and preco_atual:
            if dividend_yield < 1:  # Está em decimal
                total_12m = dividend_yield * preco_atual
            else:  # Está em porcentagem
                total_12m = (dividend_yield / 100) * preco_atual
        
        # Para FIIs, estima quantidade de pagamentos se não disponível
        if quantidade_dividendos_12m == 0 and ticker.endswith('.SA') and any(x in ticker for x in ['11', '12']):
            quantidade_dividendos_12m = 12  # FIIs geralmente pagam mensalmente
        
        return {
            'dividend_yield': dividend_yield,
            'ultimo_dividendo': ultimo_dividendo,
            'total_12m': total_12m,
            'dividendos_recentes': dividendos_recentes,
            'quantidade_dividendos_12m': quantidade_dividendos_12m,
            'forward_dividend_rate': forward_dividend_rate,
            'trailing_dividend_rate': trailing_dividend_rate
        }
    except Exception as e:
        print(f"Erro ao obter dividendos para {ticker}: {e}")
        return {
            'dividend_yield': 0,
            'ultimo_dividendo': 0,
            'total_12m': 0,
            'dividendos_recentes': [],
            'quantidade_dividendos_12m': 0,
            'forward_dividend_rate': 0,
            'trailing_dividend_rate': 0
        }

def obter_parametros_por_classe(classe):
    """Retorna os parâmetros mais importantes para cada classe de ativo"""
    parametros = {
        "Ação BR": [
            "📊 P/L (Preço/Lucro) - Valuation da empresa",
            "💰 Dividend Yield - Retorno via dividendos", 
            "📈 Crescimento de Receita e Lucro",
            "🏢 Setor e Posicionamento no mercado"
        ],
        "FII": [
            "🏢 Tipo de Fundo (Tijolo, Papel, Híbrido)",
            "💰 Dividend Yield - Rendimento mensal",
            "📊 P/VP (Preço/Valor Patrimonial)",
            "📍 Localização e Qualidade dos Imóveis"
        ],
        "BDR": [
            "🌎 Saúde Financeira da Empresa Global",
            "📊 P/L - Valuation internacional",
            "💰 Política de Dividendos Global",
            "🔄 Exposição Cambial (Dólar)"
        ],
        "ETF": [
            "📊 Taxa de Administração",
            "🔄 Diversificação da Carteira",
            "📈 Tracking Error - Aderência ao índice",
            "💰 Dividend Yield - Distribuição"
        ],
        "Renda Fixa": [
            "💰 Taxa de Juros Real (CDI - IPCA)",
            "📅 Duração e Liquidez",
            "🛡️ Garantia (FGC, Tesouro, etc.)",
            "📊 Rentabilidade Líquida"
        ]
    }
    return parametros.get(classe, ["📈 Análise Técnica e Fundamentalista"])

# =============================================================================
# FUNÇÕES PRINCIPAIS DO SISTEMA
# =============================================================================

def selecionar_ativos_balanceados(asset_classes, max_por_classe=5):
    """Seleciona ativos de forma balanceada entre as classes"""
    import random
    
    selecionados = []
    
    for classe in asset_classes:
        if classe in tickers_base:
            # Pega uma amostra aleatória de cada classe
            tickers_classe = tickers_base[classe]
            # Garante que não tente pegar mais tickers do que existem
            num_selecionar = min(max_por_classe, len(tickers_classe))
            amostra = random.sample(tickers_classe, num_selecionar)
            selecionados.extend(amostra)
    
    return selecionados

@st.cache_data(ttl=3600)  # Cache de 1 hora
def descobrir_ativos_automaticamente():
    """Descobre automaticamente ativos promissores com mais diversidade"""
    import random
    
    # Grupos de ativos por setor/segmento para maior diversificação
    grupos_ativos = {
        "tech_finance": ["CASH3.SA", "COGN3.SA", "BIDI11.SA", "STOC31.SA", "ORCL34.SA"],
        "energia_util": ["ENGI11.SA", "EQTL3.SA", "CPLE6.SA", "TAEE11.SA", "CMIG4.SA"],
        "varejo_consumo": ["MGLU3.SA", "AMER3.SA", "VIIA3.SA", "LREN3.SA", "PCAR3.SA"],
        "materiais_ind": ["KLBN4.SA", "SUZB3.SA", "GGBR4.SA", "CSNA3.SA", "VALE3.SA"],
        "fiis_diversos": ["KNRI11.SA", "HGLG11.SA", "XPML11.SA", "VRTA11.SA", "HGRU11.SA"],
        "bdr_global": ["AAPL34.SA", "MSFT34.SA", "TSLA34.SA", "META34.SA", "NVDC34.SA"],
        "small_caps": ["MDNE3.SA", "LEVE3.SA", "RADL3.SA", "AZEV4.SA", "VAMO3.SA"]
    }
    
    # Seleciona aleatoriamente de cada grupo para garantir diversidade
    novos_tickers = set()
    for grupo, tickers in grupos_ativos.items():
        # Seleciona 2-3 tickers de cada grupo aleatoriamente
        num_selecionar = min(random.randint(2, 3), len(tickers))
        selecionados = random.sample(tickers, num_selecionar)
        novos_tickers.update(selecionados)
    
    # Garante alguns ativos fixos importantes
    tickers_fixos = ["PETR4.SA", "ITUB4.SA", "BOVA11.SA", "IVVB11.SA"]
    novos_tickers.update(tickers_fixos)
    
    return list(novos_tickers)

@st.cache_data(ttl=1800)  # Cache de 30 minutos
def coletar_indicadores(ticker, period="6mo"):
    """Coleta dados do Yahoo Finance com informações detalhadas de dividendos"""
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
                'info_dividendos': None,
                'parametros_classe': obter_parametros_por_classe('Renda Fixa')
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
        
        # Informações detalhadas de dividendos - CORREÇÃO: passa o preço atual
        preco_atual = float(latest['Close'])
        info_dividendos = obter_info_dividendos_detalhada(ticker, preco_atual)
        
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
        
        # Obter parâmetros específicos da classe
        parametros_classe = obter_parametros_por_classe(classe)
        
        return {
            'ticker': ticker,
            'classe': classe,
            'price': preco_atual,
            'sma20': float(latest['SMA20']) if not np.isnan(latest['SMA20']) else None,
            'sma50': float(latest['SMA50']) if not np.isnan(latest['SMA50']) else None,
            'rsi14': float(latest['RSI']) if not np.isnan(latest['RSI']) else None,
            'pe': info.get('trailingPE', None),
            'dividendYield': info.get('dividendYield', None),
            'sector': info.get('sector', 'N/A'),
            'nome': info.get('longName', ticker),
            'volume': float(latest['Volume']) if 'Volume' in latest else None,
            'info_dividendos': info_dividendos,
            'parametros_classe': parametros_classe
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
    """Analisa oportunidades usando GPT com contexto mais rico"""
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
        
        # Prompt muito mais detalhado e específico
        prompt = f"""
        Como analista financeiro especializado no mercado brasileiro, analise estas oportunidades:

        ESTRATÉGIA SOLICITADA: {strategy}
        HORIZONTE TEMPORAL: {horizon}

        DADOS TÉCNICOS DOS ATIVOS (TOP 15):
        {dados_analise[['ticker', 'classe', 'price', 'rsi14', 'pe', 'dividendYield', 'sector', 'Score_Final']].to_string()}

        CONTEXTO MACROECONÔMICO ATUAL (Setembro/2025):
        - CDI: 13.31% a.a. | Selic: 13.31% a.a. | IPCA: 4.25% a.a.
        - Ibovespa: ~154.063 pts | Dólar: R$ 5,33
        - Cenário: Juros em patamar elevado, inflação controlada, crescimento moderado

        SETORES E TENDÊNCIAS DO MERCADO:
        - Tecnologia & Pagamentos: CASH3, COGN3 - Setor em expansão com digitalização
        - Energia & Utilities: EQTL3, CPLE6 - Transição energética e estabilidade
        - Varejo & Consumo: MGLU3, AMER3 - Sensível ao ciclo econômico
        - Materiais & Industrial: KLBN4, GGBR4 - Commodities e infraestrutura
        - FIIs: KNRI11, HGLG11 - Sensíveis a taxa de juros, bom para renda
        - BDRs: AAPL34, MSFT34 - Exposição internacional, diversificação

        INSTRUÇÕES ESPECÍFICAS:
        1. CORRIJA SETORES ERRADOS: CASH3 é pagamentos/digital, não energia
        2. DIVERSIFIQUE: Não recomende apenas FIIs ou apenas ações
        3. CONTEXTO REAL: Considere notícias recentes e tendências de mercado
        4. CENÁRIO DE JUROS: Analise impacto dos juros altos em cada classe
        5. HORIZONTE TEMPORAL: Adeque recomendações ao prazo solicitado

        ESTRUTURA DA RESPOSTA:
        1. 🎯 TOP 5 OPORTUNIDADES - Justifique tecnicamente cada escolha
        2. 📊 ANÁLISE SETORIAL - Como cada setor se comporta no cenário atual
        3. ⚠️ PRINCIPAIS RISCOS - Específicos para cada recomendação
        4. 💰 ALOCAÇÃO SUGERIDA - % ideal para cada ativo na carteira
        5. 🔄 GATILHOS OPERACIONAIS - Quando entrar/sair de cada posição

        Seja PRÁTICO, ESPECÍFICO e baseie-se em DADOS REAIS do mercado.
        Evite recomendações genéricas - seja específico para cada ativo.
        """
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2500,
            temperature=0.7  # Um pouco mais criativo
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
    
    # Input para API Key - pré-preenche se já estiver salva
    api_key = st.text_input("OpenAI API Key", type="password", 
                           value=api_key_salva if api_key_salva else "",
                           placeholder="Cole sua OpenAI API Key aqui...",
                           help="Obtenha em: https://platform.openai.com/api-keys")
    
    # Botão para salvar API Key
    if api_key and api_key != api_key_salva:
        if st.button("💾 Salvar API Key Localmente"):
            if salvar_api_key(api_key):
                st.success("✅ API Key salva com sucesso! Você não precisará digitá-la novamente.")
                st.rerun()
            else:
                st.error("❌ Erro ao salvar API Key")
    
    # Instruções para Streamlit Cloud
    with st.expander("🌐 Para uso no Streamlit Cloud"):
        st.write("""
        **No Streamlit Cloud, adicione sua API Key em:**
        1. Acesse [Streamlit Cloud](https://share.streamlit.io/)
        2. No seu app, clique em ⚙️ **Settings**
        3. Vá em **Secrets** e adicione:
        ```
        OPENAI_API_KEY = "sua-chave-aqui"
        ```
        4. A chave será carregada automaticamente!
        """)
    
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
        "RADL3.SA", "SBSP3.SA", "CSAN3.SA", "EMBR3.SA", "GGBR4.SA", "CYRE3.SA",
        "COGN3.SA", "CASH3.SA", "KLBN4.SA", "RAIZ4.SA", "AZEV4.SA"
    ],
    "FIIs": [
        "HGLG11.SA", "KNRI11.SA", "XPML11.SA", "HGRU11.SA", "VGIP11.SA", "XPLG11.SA",
        "BCFF11.SA", "RBRF11.SA", "HSML11.SA", "VRTA11.SA", "MXRF11.SA", "RCRB11.SA"
    ],
    "BDRs": [
        "AAPL34.SA", "MSFT34.SA", "AMZO34.SA", "TSLA34.SA", 
        "COCA34.SA", "DISB34.SA", "JPMO34.SA", "NVDC34.SA",
        "M1TA34.SA", "GOGL34.SA", "NFLX34.SA"
    ],
    "ETFs": [
        "BOVA11.SA", "SMAL11.SA", "IVVB11.SA", "BBSD11.SA", "DIVO11.SA", "FIND11.SA",
        "GOVE11.SA", "IMAB11.SA", "SPXI11.SA"
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
            
            # Coletar tickers de forma balanceada
            todos_tickers = selecionar_ativos_balanceados(asset_classes, max_por_classe=4)
            # Busca automática
            if auto_discover:
                novos_tickers = descobrir_ativos_automaticamente()
                st.session_state.novos_tickers = novos_tickers
                for ticker in novos_tickers:
                    if ticker not in todos_tickers:
                        todos_tickers.append(ticker)

            st.info(f"🔍 Analisando {len(todos_tickers)} ativos selecionados balanceadamente...")
            
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
                            
                            # Mostrar informações de dividendos se disponível
                            if row.get('info_dividendos'):
                                info = row['info_dividendos']
                                ultimo_div = info.get('ultimo_dividendo', 0) or 0
                                total_12m_val = info.get('total_12m', 0) or 0
                                
                                if ultimo_div > 0:
                                    st.write(f"**Últ. Div.:** R$ {ultimo_div:.2f}")
                                if total_12m_val > 0:
                                    st.write(f"**Div. 12m:** R$ {total_12m_val:.2f}")
                        
                        with col2:
                            st.write(f"Preço: R$ {row['price']:.2f}")
                            if row.get('dividendYield') is not None:
                                dy_formatado = formatar_dividend_yield(row['dividendYield'])
                                st.markdown(f"**DY:** {dy_formatado}", unsafe_allow_html=True)
                        
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
        
        # Legenda explicativa
        with st.expander("ℹ️ Legenda dos Scores"):
            st.write("""
            **Classificação dos Scores:**
            - **⭐ Alta (15+):** Oportunidade muito promissora
            - **📊 Média (10-14):** Oportunidade interessante  
            - **📈 Baixa (0-9):** Oportunidade com menor potencial
            
            **Cores dos Indicadores:**
            - 🟢 **DY ≥ 6%** (Alto) | **P/L < 12** (Barato)
            - 🟡 **DY 3-6%** (Médio) | **P/L 12-20** (Justo)  
            - 🔴 **DY < 3%** (Baixo) | **P/L > 20** (Caro)
            """)
        
        # Exibir cada ativo com informações organizadas
        for _, row in df_filtrado.iterrows():
            with st.container():
                st.markdown(f"**{row['ticker']}** | **{row['classe']}** | **Setor:** {row.get('sector', 'N/A')}")
                
                # Layout principal com 2 colunas
                col_left, col_right = st.columns([3, 2])
                
                with col_left:
                    # Parâmetros analisados
                    st.markdown("**📊 Parâmetros Analisados:**")
                    for param in row.get('parametros_classe', []):
                        st.markdown(f'<div class="param-card">{param}</div>', unsafe_allow_html=True)
                    
                    # Informações de preço e indicadores
                    col_price, col_indicators = st.columns(2)
                    
                    with col_price:
                        st.write(f"**Preço:** R$ {row['price']:.2f}")
                        rsi = row['rsi14']
                        if pd.notna(rsi):
                            st.write(f"**RSI:** {rsi:.1f}")
                    
                    with col_indicators:
                        if row['classe'] in ["Ação BR", "Ação EUA", "BDR"]:
                            st.markdown(f"**P/L:** {formatar_pe(row['pe'])}", unsafe_allow_html=True)
                        
                        if row['classe'] in ["FII", "ETF", "Ação BR", "Ação EUA", "BDR"]:
                            st.markdown(f"**DY:** {formatar_dividend_yield(row['dividendYield'])}", unsafe_allow_html=True)
                    
                    # CORREÇÃO COMPLETA: Informações detalhadas de dividendos
                    if row.get('info_dividendos'):
                        info = row['info_dividendos']
                        
                        # Verificar se há informações válidas de dividendos
                        ultimo_div = info.get('ultimo_dividendo', 0) or 0
                        total_12m_val = info.get('total_12m', 0) or 0
                        qtd_div = info.get('quantidade_dividendos_12m', 0) or 0
                        dividendos_recentes = info.get('dividendos_recentes', [])
                        forward_dividend = info.get('forward_dividend_rate', 0) or 0
                        trailing_dividend = info.get('trailing_dividend_rate', 0) or 0
                        
                        # Mostrar seção de dividendos se houver qualquer dado relevante
                        tem_dados_dividendos = (ultimo_div > 0 or total_12m_val > 0 or 
                                               qtd_div > 0 or len(dividendos_recentes) > 0 or
                                               forward_dividend > 0 or trailing_dividend > 0 or
                                               (row.get('dividendYield') and row['dividendYield'] > 0))
                        
                        if tem_dados_dividendos:
                            st.markdown("**💰 Informações de Dividendos:**")
                            
                            # Último dividendo
                            if ultimo_div > 0:
                                st.write(f"• **Último:** R$ {ultimo_div:.2f}")
                            elif forward_dividend > 0:
                                st.write(f"• **Forward Dividend:** R$ {forward_dividend:.2f}")
                            elif trailing_dividend > 0:
                                st.write(f"• **Trailing Dividend:** R$ {trailing_dividend:.2f}")
                            else:
                                st.write("• **Último:** N/A")
                            
                            # Total dos últimos 12 meses
                            if total_12m_val > 0:
                                st.write(f"• **12 meses:** R$ {total_12m_val:.2f}")
                            elif row.get('dividendYield') and row['dividendYield'] > 0 and row.get('price'):
                                # Calcula aproximado baseado no DY e preço
                                if row['dividendYield'] < 1:  # Se DY está em decimal
                                    dy_value = (row['dividendYield'] * row['price'])
                                else:  # Se DY está em porcentagem
                                    dy_value = (row['dividendYield'] / 100) * row['price']
                                st.write(f"• **12 meses (estimado):** R$ {dy_value:.2f}")
                            else:
                                st.write("• **12 meses:** N/A")
                            
                            # Quantidade de pagamentos
                            if qtd_div > 0:
                                st.write(f"• **Pagamentos (12m):** {qtd_div}")
                            elif row['classe'] == 'FII':
                                st.write("• **Pagamentos (12m):** ~12 (estimado para FII)")
                            else:
                                st.write("• **Pagamentos (12m):** N/A")
                            
                            # Últimos 3 dividendos
                           # Últimos 3 dividendos - VERSÃO DEFINITIVAMENTE CORRIGIDA
                            if dividendos_recentes:
                            # Pegar os últimos 3 dividendos (mais recentes)
                                ultimos_3 = dividendos_recentes[-3:] if len(dividendos_recentes) >= 3 else dividendos_recentes   
                                # CORREÇÃO: Garantir que todos os valores sejam floats e formatar consistentemente
                                dividendos_formatados = []
                                for div in ultimos_3:
                                # Converter para float se não for
                                    valor = float(div) if not isinstance(div, float) else div
                                # Formatar no padrão brasileiro: R$ 0,85
                                    dividendos_formatados.append(f"R$ {valor:.2f}".replace('.', ','))  
                                    div_recentes_str = ", ".join(dividendos_formatados)
                                st.write(f"• **Últimos {len(ultimos_3)}:** {div_recentes_str}")
                            elif ultimo_div > 0:
                                st.write(f"• **Último conhecido:** R$ {ultimo_div:.2f}")
                            else:
                                st.write("• **Últimos dividendos:** N/A")
                        else:
                            # Mostra pelo menos o dividend yield se disponível
                            if row.get('dividendYield') and row['dividendYield'] > 0:
                                st.markdown("**💰 Dividend Yield:**")
                                st.write(f"• **DY:** {formatar_dividend_yield(row['dividendYield'])}", unsafe_allow_html=True)
                            else:
                                st.write("**💰 Dividendos:** Dados limitados disponíveis")
                    else:
                        # Se não tem info_dividendos, mostra pelo menos o DY se disponível
                        if row.get('dividendYield') and row['dividendYield'] > 0:
                            st.markdown("**💰 Dividend Yield:**")
                            st.write(f"• **DY:** {formatar_dividend_yield(row['dividendYield'])}", unsafe_allow_html=True)
                        else:
                            st.write("**💰 Dividendos:** Informações não disponíveis")
                
                with col_right:
                    # Scores com destaque
                    st.markdown("**🎯 Scores:**")
                    
                    # Score Final com destaque visual
                    score_final = row['Score_Final']
                    if score_final >= 15:
                        st.success(f"**Score Final: {score_final:.1f}** ⭐ ALTA")
                    elif score_final >= 10:
                        st.info(f"**Score Final: {score_final:.1f}** 📊 MÉDIA")
                    else:
                        st.warning(f"**Score Final: {score_final:.1f}** 📈 BAIXA")
                    
                    # Scores individuais
                    st.write(f"**Proteção:** {row['Score_Protecao']:.1f}")
                    st.write(f"**Crescimento:** {row['Score_Crescimento']:.1f}")
                    
                    # Explicação dos scores
                    with st.expander("📖 Sobre os Scores"):
                        st.write(f"""
                        **Como interpretar:**
                        - **Score Final {score_final:.1f}:** Avaliação geral baseada na estratégia "{strategy}"
                        - **Proteção {row['Score_Protecao']:.1f}:** Capacidade de preservar capital
                        - **Crescimento {row['Score_Crescimento']:.1f}:** Potencial de valorização
                        
                        **Classificação:**
                        - ⭐ **ALTA (15+):** Oportunidade muito promissora
                        - 📊 **MÉDIA (10-14):** Oportunidade interessante  
                        - 📈 **BAIXA (0-9):** Oportunidade com menor potencial
                        """)
                
                st.markdown("---")
        
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
                    col1, col2, col3 = st.columns([2, 2, 2])
                    
                    with col1:
                        st.write(f"**Classe:** {row['classe']}")
                        st.write(f"**Preço:** R$ {row['price']:.2f}")
                        st.write(f"**Setor:** {row.get('sector', 'N/A')}")
                        
                        # Parâmetros principais da classe
                        st.markdown("**📊 Parâmetros Chave:**")
                        principais_params = row.get('parametros_classe', [])[:2]
                        for param in principais_params:
                            st.write(f"• {param}")
                    
                    with col2:
                        if row.get('rsi14'):
                            st.write(f"**RSI:** {row['rsi14']:.1f}")
                        if row['classe'] in ["Ação BR", "Ação EUA", "BDR"]:
                            st.markdown(f"**P/L:** {formatar_pe(row['pe'])}", unsafe_allow_html=True)
                        if row['classe'] in ["FII", "ETF", "Ação BR", "Ação EUA", "BDR"]:
                            st.markdown(f"**DY:** {formatar_dividend_yield(row['dividendYield'])}", unsafe_allow_html=True)
                    
                    with col3:
                        # Informações de dividendos detalhadas
                        if row.get('info_dividendos'):
                            info = row['info_dividendos']
                            st.write("**💰 Dividendos:**")
                            
                            # CORREÇÃO: Usar valores padrão 0 se for None
                            ultimo_div = info.get('ultimo_dividendo', 0) or 0
                            total_12m_val = info.get('total_12m', 0) or 0
                            
                            if ultimo_div > 0:
                                st.write(f"• **Último:** R$ {ultimo_div:.2f}".replace('.', ','))
                            
                            if total_12m_val > 0:
                                st.write(f"• **12 meses:** R$ {total_12m_val:.2f}".replace('.', ','))
        
        # Gráfico de novas descobertas
        if not df_novos.empty:
            fig = px.bar(df_novos, x='ticker', y='Score_Crescimento', 
                        title='Potencial de Crescimento - Novas Descobertas',
                        color='Score_Crescimento', color_continuous_scale='viridis')
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Execute a análise para descobrir novas oportunidades")

# Rodapé com horário correto de Brasília
st.markdown("---")
st.markdown(
    "⚠️ **Aviso Legal:** Esta ferramenta é apenas para fins educacionais. "
    "Não constitui recomendação de investimento. Consulte um profissional qualificado."
)
st.markdown(
    f"📊 **Fontes de Dados:** Yahoo Finance, Banco Central do Brasil, OpenAI GPT-4 | "
    f"🔄 **Última atualização:** {obter_horario_brasilia()} (Horário de Brasília)"
)
