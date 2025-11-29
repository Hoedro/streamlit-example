import streamlit as st
import pandas as pd
import feedparser
import sqlite3
import json
import os
import time
import requests
import re
import concurrent.futures
import asyncio
import html
from datetime import datetime, timedelta, timezone
from io import BytesIO

# --- Imports Opcionais (Try/Except para não quebrar se faltar algo) ---
try:
    from deep_translator import GoogleTranslator
    TRANSLATOR_AVAILABLE = True
except ImportError:
    TRANSLATOR_AVAILABLE = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

try:
    from telethon import TelegramClient
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False

try:
    import cloudscraper
    CLOUDSCRAPER_AVAILABLE = True
except ImportError:
    CLOUDSCRAPER_AVAILABLE = False

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# =============================================================================
st.set_page_config(
    page_title="OSINT DAMÁSIO V19",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilo CSS Personalizado
st.markdown("""
<style>
    .stButton>button {width: 100%; border-radius: 5px; height: 3em;}
    .reportview-container {background: #0e1117;}
    h1 {color: #00ff41;}
    .stAlert {background-color: #1c2e2e; color: #eee;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# GESTÃO DE ESTADO (SESSION STATE)
# =============================================================================
if 'df_resultados' not in st.session_state:
    st.session_state['df_resultados'] = pd.DataFrame()
if 'termos_busca' not in st.session_state:
    st.session_state['termos_busca'] = []
if 'config' not in st.session_state:
    # Tenta carregar config.json, senão usa padrão
    if os.path.exists('config.json'):
        with open('config.json', 'r') as f:
            st.session_state['config'] = json.load(f)
    else:
        st.session_state['config'] = {
            "rss_feeds": {"ZONA: GERAL": {"CNN": "http://rss.cnn.com/rss/edition.rss"}},
            "telegram_channels": [],
            "gemini_api_key": ""
        }

DB_MEMORIA = 'memoria_consciente.db'

# =============================================================================
# FUNÇÕES DE LÓGICA (BACKEND)
# =============================================================================

def iniciar_memoria_consciente():
    conn = sqlite3.connect(DB_MEMORIA)
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS licoes (id INTEGER PRIMARY KEY AUTOINCREMENT, data_registro TEXT, termos_chave TEXT, erro_cometido TEXT, licao_aprendida TEXT, impacto INTEGER DEFAULT 5, veracidade TEXT DEFAULT 'Fato Confirmado', contexto TEXT DEFAULT 'Geral')''')
    cursor.execute('''CREATE TABLE IF NOT EXISTS diario_bordo (id INTEGER PRIMARY KEY AUTOINCREMENT, data_evento TEXT, tipo_analise TEXT, resumo_gerado TEXT)''')
    conn.commit()
    conn.close()

def gravar_memoria(termos, erro, licao, impacto, veracidade, contexto):
    conn = sqlite3.connect(DB_MEMORIA)
    cursor = conn.cursor()
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    cursor.execute("INSERT INTO licoes (data_registro, termos_chave, erro_cometido, licao_aprendida, impacto, veracidade, contexto) VALUES (?, ?, ?, ?, ?, ?, ?)",
                   (ts, termos, erro, licao, impacto, veracidade, contexto))
    conn.commit()
    conn.close()
    return "Memória consolidada com sucesso."

def consultar_memoria(termos_lista):
    conn = sqlite3.connect(DB_MEMORIA)
    cursor = conn.cursor()
    licoes = []
    
    # 1. Busca Global (Alto Impacto)
    cursor.execute("SELECT erro_cometido, licao_aprendida, impacto, contexto FROM licoes WHERE impacto >= 8")
    licoes.extend(cursor.fetchall())
    
    # 2. Busca Específica
    for t in termos_lista:
        cursor.execute("SELECT erro_cometido, licao_aprendida, impacto, contexto FROM licoes WHERE termos_chave LIKE ?", (f'%{t}%',))
        licoes.extend(cursor.fetchall())
    
    conn.close()
    
    if not licoes: return ""
    
    # Remove duplicados e formata
    unique = list(set(licoes))
    txt = "\n\n### 🧠 MEMÓRIA DO SISTEMA (PREMISSAS):\n"
    for erro, licao, imp, ctx in unique:
        txt += f"- [{ctx}] (Imp:{imp}): {licao}\n"
    return txt

def chamar_gemini(system_prompt, user_prompt, api_key):
    if not api_key: return "⚠️ ERRO: API Key não configurada."
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": user_prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "generationConfig": {"temperature": 0.4}
    }
    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'}, timeout=60)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        return f"Erro API ({response.status_code}): {response.text}"
    except Exception as e: return f"Erro Rede: {e}"

def executar_enxame(prompt_usuario, dados_contexto, api_key):
    """Executa 3 agentes em paralelo e integra os resultados."""
    if not api_key: return "Configure a API Key na barra lateral."
    
    input_comum = f"{prompt_usuario}\n\n[DADOS]:\n{dados_contexto[:15000]}" # Limite de caracteres para segurança
    
    prompts = {
        "Militar": "Tu és um Especialista Militar. Foca em movimentações, armas, logística e ameaças cinéticas. Sê técnico.",
        "Politico": "Tu és um Diplomata. Foca em tensões políticas, discursos, eleições e geopolítica.",
        "Social": "Tu és um Analista Social. Foca em impacto na população, economia, greves e infraestruturas civis."
    }
    
    resultados = {}
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_agent = {executor.submit(chamar_gemini, p_sys, input_comum, api_key): nome for nome, p_sys in prompts.items()}
        for future in concurrent.futures.as_completed(future_to_agent):
            nome = future_to_agent[future]
            try:
                resultados[nome] = future.result()
            except Exception as e:
                resultados[nome] = f"Erro: {e}"
                
    # Agente Integrador
    prompt_integrador = """
    Tu és o Diretor de Inteligência. Recebeste 3 relatórios.
    Sintetiza tudo num relatório final coerente.
    Resolve contradições assumindo o pior cenário (Prudência).
    Estrutura: SUMÁRIO, AMEAÇAS, CENÁRIOS.
    """
    input_integracao = "\n".join([f"[{k.upper()}]: {v}" for k,v in resultados.items()])
    
    return chamar_gemini(prompt_integrador, input_integracao, api_key)

def obter_rss(url, termos):
    """Baixa e filtra RSS"""
    items = []
    try:
        # Tenta requests normal
        resp = requests.get(url, timeout=5)
        content = resp.text
    except:
        # Fallback Cloudscraper
        if CLOUDSCRAPER_AVAILABLE:
            try:
                scraper = cloudscraper.create_scraper()
                content = scraper.get(url).text
            except: return []
        else: return []

    feed = feedparser.parse(content)
    for entry in feed.entries[:10]: # Limite por feed
        texto = (entry.get('title', '') + " " + entry.get('summary', '')).lower()
        
        # Filtro (Se termo for "*" traz tudo)
        match = False
        if "*" in termos:
            match = True
        else:
            for t in termos:
                if t.lower() in texto:
                    match = True
                    break
        
        if match:
            # Tradução de Título se necessário
            titulo = entry.get('title', 'Sem Titulo')
            if TRANSLATOR_AVAILABLE and not titulo.isascii(): # Heuristica simples
                try: titulo = GoogleTranslator(source='auto', target='pt').translate(titulo)
                except: pass
                
            items.append({
                'Data': entry.get('published', datetime.now().strftime('%Y-%m-%d')),
                'Fonte': feed.feed.get('title', 'RSS'),
                'Titulo': titulo,
                'Resumo': entry.get('summary', '')[:300],
                'Link': entry.get('link', '#')
            })
    return items

def gerar_pdf(df, titulo):
    if not REPORTLAB_AVAILABLE: return None
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = [Paragraph(titulo, styles['Title']), Spacer(1, 12)]
    
    for _, row in df.iterrows():
        text = f"<b>{row['Fonte']}</b>: <a href='{row['Link']}'>{row['Titulo']}</a><br/>{row['Resumo']}"
        story.append(Paragraph(text, styles['Normal']))
        story.append(Spacer(1, 12))
        
    doc.build(story)
    buffer.seek(0)
    return buffer

# =============================================================================
# INTERFACE GRÁFICA (STREAMLIT)
# =============================================================================

# --- SIDEBAR (Configurações) ---
with st.sidebar:
    st.header("⚙️ Configuração")
    
    # API Key
    api_key_input = st.text_input("Gemini API Key", value=st.session_state['config'].get('gemini_api_key', ''), type="password")
    if api_key_input != st.session_state['config'].get('gemini_api_key', ''):
        st.session_state['config']['gemini_api_key'] = api_key_input
        # Salvar config se quiseres persistência local
        # with open('config.json', 'w') as f: json.dump(st.session_state['config'], f)

    st.markdown("---")
    st.subheader("Fontes Ativas")
    
    # Seleção de RSS
    feeds_config = st.session_state['config'].get('rss_feeds', {})
    feeds_selecionados = []
    
    # Expander para RSS
    with st.expander("📡 Feeds RSS", expanded=False):
        for zona, feeds in feeds_config.items():
            st.markdown(f"**{zona}**")
            for nome, url in feeds.items():
                if st.checkbox(f"{nome}", value=True, key=f"rss_{nome}"):
                    feeds_selecionados.append((nome, url))

    # Expander para Telegram
    canais_selecionados = []
    if TELEGRAM_AVAILABLE:
        tg_channels = st.session_state['config'].get('telegram_channels', [])
        with st.expander("✈️ Telegram", expanded=False):
            st.info("Requer 'session' configurada no servidor.")
            for canal in tg_channels:
                if st.checkbox(f"@{canal}", value=True, key=f"tg_{canal}"):
                    canais_selecionados.append(canal)

# --- CORPO PRINCIPAL ---

st.title("👁️ OSINT COLECTOR - DAMÁSIO WEB")
st.markdown("*Sistema de Inteligência Multi-Fonte com Análise Cognitiva*")

# 1. INPUT DE BUSCA
col1, col2 = st.columns([3, 1])
with col1:
    termos_input = st.text_input("Alvos (separados por vírgula ou '*' para tudo)", "Lajes, Açores, Submarino")
with col2:
    st.write("") # Espaçamento
    st.write("")
    btn_scan = st.button("🚀 INICIAR SCAN", type="primary")

# 2. LÓGICA DE SCAN
if btn_scan:
    iniciar_memoria_consciente()
    termos = [t.strip() for t in termos_input.split(',')]
    st.session_state['termos_busca'] = termos
    
    resultados = []
    status_bar = st.status("A recolher inteligência...", expanded=True)
    
    # RSS Scan
    status_bar.write("📡 A ler RSS Feeds...")
    total_rss = len(feeds_selecionados)
    
    # ThreadPool para RSS (Scan Paralelo)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(obter_rss, url, termos): nome for nome, url in feeds_selecionados}
        for future in concurrent.futures.as_completed(future_to_url):
            try:
                data = future.result()
                resultados.extend(data)
            except Exception as e:
                st.error(f"Erro num feed: {e}")
    
    # Telegram Scan (Simulado/Simplificado para Web)
    if TELEGRAM_AVAILABLE and canais_selecionados:
        status_bar.write("✈️ A ler Telegram (Async)...")
        # Nota: Telegram em Streamlit requer loop async cuidado. 
        # Aqui simplificamos. Se tiveres credenciais, usa o client.
        pass 

    status_bar.update(label="Scan Concluído!", state="complete", expanded=False)
    
    if resultados:
        df = pd.DataFrame(resultados)
        # Ordenar por data (assumindo formato string ISO ou similar, ajustar conforme necessidade)
        st.session_state['df_resultados'] = df
        st.success(f"Encontrados {len(df)} resultados.")
    else:
        st.warning("Nenhum resultado encontrado.")

# 3. VISUALIZAÇÃO DE RESULTADOS
if not st.session_state['df_resultados'].empty:
    df = st.session_state['df_resultados']
    
    # Dataframe Interativo
    st.dataframe(
        df,
        column_config={
            "Link": st.column_config.LinkColumn("Link Original"),
            "Resumo": st.column_config.TextColumn("Snippet", width="medium"),
        },
        use_container_width=True,
        hide_index=True
    )

    # 4. ÁREA DE INTELIGÊNCIA ARTIFICIAL
    st.markdown("---")
    st.subheader("🧠 Análise de Inteligência (IA)")
    
    col_ia1, col_ia2 = st.columns(2)
    
    with col_ia1:
        st.markdown("### Análise Tática")
        st.caption("Resumo rápido de ameaças e eventos.")
        if st.button("Executar Análise Tática"):
            with st.spinner("A consultar o Enxame..."):
                memoria = consultar_memoria(st.session_state['termos_busca'])
                dados_texto = df.to_string()
                
                prompt = f"{memoria}\nAnalise os dados acima. Identifique ameaças imediatas e eventos críticos. Seja breve."
                res = chamar_gemini("Analista de Defesa", prompt, st.session_state['config']['gemini_api_key'])
                st.markdown(res)
                
    with col_ia2:
        st.markdown("### Análise Estratégica")
        st.caption("Visão profunda (Militar, Política, Social).")
        if st.button("Executar Enxame (3 Agentes)"):
            with st.spinner("A ativar Agentes Especialistas..."):
                memoria = consultar_memoria(st.session_state['termos_busca'])
                dados_texto = df[['Fonte', 'Titulo', 'Resumo']].to_string()
                
                prompt = f"{memoria}\nProduz um relatório de inteligência estratégica baseado nos dados."
                res = executar_enxame(prompt, dados_texto, st.session_state['config']['gemini_api_key'])
                st.info("Relatório Integrado:")
                st.markdown(res)

    # 5. EXPORTAÇÃO E MEMÓRIA
    st.markdown("---")
    col_exp, col_mem = st.columns(2)
    
    with col_exp:
        st.subheader("📂 Exportação")
        # CSV
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Baixar CSV", csv, "osint_data.csv", "text/csv")
        
        # PDF
        if REPORTLAB_AVAILABLE:
            pdf_buffer = gerar_pdf(df, "Relatório OSINT")
            if pdf_buffer:
                st.download_button("📄 Baixar PDF", pdf_buffer, "relatorio.pdf", "application/pdf")

    with col_mem:
        st.subheader("💉 Gestor de Memória")
        with st.expander("Ensinar o Sistema (Feedback Loop)"):
            with st.form("form_memoria"):
                erro = st.text_input("Erro Cometido (O que a IA pensou errado?)")
                licao = st.text_input("Lição (Qual a verdade a assumir?)")
                impacto = st.slider("Impacto", 1, 10, 5)
                contexto = st.selectbox("Contexto", ["Militar", "Político", "Geral"])
                
                if st.form_submit_button("Gravar na Consciência"):
                    msg = gravar_memoria(str(st.session_state['termos_busca']), erro, licao, impacto, "Fato", contexto)
                    st.success(msg)

# Rodapé
st.markdown("---")
st.caption("Sistema OSINT V19 - Streamlit Edition | Programado por Pedro Horta")
