import os

# ***********************************************************************
# -------------------OCULTAR OS AVISOS DO TENSORFLOW ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# ***********************************************************************

import json
import logging
import time
import csv
from urllib.parse import urlparse
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup

# Importando o html2text para redução de tokens
import html2text

# Importando o DrissionPage
from DrissionPage import ChromiumPage, ChromiumOptions

load_dotenv()

logging.getLogger('tensorflow').setLevel(logging.FATAL)

# --- CONFIGURAÇÕES DE PERSISTÊNCIA E DATASET ---
CACHE_FILE_ALL = "processed_links_all.json"
CACHE_FILE_FOUND = "processed_links_found.json"
CHECKPOINT_FILE = "dataset_checkpoint.jsonl"
CLICK_MEMORY = {}

def carregar_cache(tipo="all"):
    arquivo = CACHE_FILE_ALL if tipo == "all" else CACHE_FILE_FOUND
    if os.path.exists(arquivo):
        try:
            with open(arquivo, 'r') as f:
                return set(json.load(f))
        except:
            return set()
    return set()

def salvar_no_cache(link, tipo="all"):
    arquivo = CACHE_FILE_ALL if tipo == "all" else CACHE_FILE_FOUND
    cache = carregar_cache(tipo)
    cache.add(link)
    with open(arquivo, 'w') as f:
        json.dump(list(cache), f)

def salvar_checkpoint(dado):
    with open(CHECKPOINT_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(dado, ensure_ascii=False) + "\n")

# ************************************************
# 1. Definir os Schemas de Saída (Pydantic)
# ************************************************
class Product(BaseModel):
    name: str = Field(description="O nome completo do produto")
    link: str = Field(description="A URL completa para a página do produto")

class ProductList(BaseModel):
    products: list[Product]

class Spec(BaseModel):
    chave: str = Field(description="O nome exato da informação solicitada (ex: Código Anatel, Marca)")
    valor: str = Field(description="O valor extraído do HTML, ou 'Não encontrado' se não existir")

class ProductDetails(BaseModel):
    detalhes: list[Spec]

# ************************************************
# 2. Funções Auxiliares e de Navegação
# ************************************************
BASE_PROFILE_PATH = r'./scraper_profile_generico'

def obter_pasta_perfil(url):
    dominio = urlparse(url).netloc
    partes_dominio = dominio.replace("www.", "").split('.')
    marketplace = str(partes_dominio) 
    return os.path.join(BASE_PROFILE_PATH, marketplace)

def limpar_html_agnostico(html_cru):
    if not html_cru:
        return ""
    soup = BeautifulSoup(html_cru, 'html.parser')
    tags_para_remover = [
        'script', 'style', 'header', 'footer', 'nav', 'aside', 
        'noscript', 'svg', 'form', 'iframe', 'button'
    ]
    for tag in soup(tags_para_remover):
        tag.decompose()
    for tag in soup.find_all(True):
        tag.attrs = {}
    return str(soup)

def html_para_markdown(html_content):
    if not html_content:
        return ""
    converter = html2text.HTML2Text()
    converter.ignore_links = False 
    converter.ignore_images = True 
    converter.ignore_emphasis = True
    return converter.handle(html_content)

def verificar_se_achou(detalhes_dict):
    falsos_positivos = ["não encontrado", "not found", "n/a", "na", "null", "none", "nenhum", "", "-", "não informado"]
    for v in detalhes_dict.values():
        val_str = str(v).strip().lower()
        if val_str not in falsos_positivos:
            return True
    return False

# ************************************************
# 3. Funções de Extração (ScrapegraphAI)
# ************************************************
def extrair_dados_estagio_1(texto_markdown, dominio, termo_busca):
    from scrapegraphai.graphs import SmartScraperGraph
    graph_config = {
        "llm": {
            "api_key": os.getenv("OPENAI_API_KEY"), 
            "model": "openai/gpt-5.4-nano",
            #"model_tokens": 128000
        },
        "verbose": False, 
        "force": True,
    }
    
    prompt = f"""
                    Target Domain: {dominio}
                    Search Term: {termo_busca}
                    Extract organic products. Discard ads. Prepend https://{dominio} to links.
                    """
    
    scraper = SmartScraperGraph(prompt=prompt, source=texto_markdown, config=graph_config, schema=ProductList)
    return scraper.run()

def extrair_dados_estagio_2(html_limpo, info_desejada):
    from scrapegraphai.graphs import SmartScraperGraph
    graph_config = {
        "llm": {
            "api_key": os.getenv("GEMINI_API_KEY_PERSONAL"),
            "model": "google_genai/gemma-3-27b-it",
            #"model_tokens": 1000000
        },
        "verbose": False, 
        "force": True,
    }
    prompt_stage2 = f"""Você é um extrator de dados. Analise o HTML desta página de produto.
    Procure especificamente por estas informações: {info_desejada}.
    Retorne uma lista de 'detalhes' com 'chave' e 'valor'."""

    scraper = SmartScraperGraph(prompt=prompt_stage2, source=html_limpo, config=graph_config, schema=ProductDetails)
    resultado = scraper.run()
    return resultado, prompt_stage2

# ************************************************
# 4. Lógica Principal Sequencial
# ************************************************
def main():
    SALVAR_NAO_ENCONTRADOS = 0 
    # FLAG DE VERIFICAÇÃO DE CACHE: "found" para considerar apenas sucessos, "all" para todos processados
    CACHE_VERIFICATION_MODE = "found"

    print("=== PIPELINE HÍBRIDO: DRISSIONPAGE + SCRAPEGRAPHAI ===")
    url = input("1. Cole a URL de pesquisa do marketplace: ").strip()
    termo_busca = input("2. Qual produto você buscou? (ex: iphone): ").strip()
    info_alvo = input("3. O que extrair de CADA produto? (ex: código anatel): ").strip()
    qtd_input = input("4. Quantos produtos deseja extrair? ").strip()
    
    try: max_produtos = int(qtd_input)
    except: max_produtos = 30
    
    dominio = urlparse(url).netloc
    
    # Seleção de cache baseada na flag
    cache_verificacao = carregar_cache(tipo=CACHE_VERIFICATION_MODE)
    
    caminho_perfil = obter_pasta_perfil(url)
    co = ChromiumOptions().set_user_data_path(caminho_perfil)
    page = ChromiumPage(co)
    
    print("\n🔍 INICIANDO ESTÁGIO 1: MAPEAMENTO DA BUSCA")
    page.get(url)
    time.sleep(5)
    
    todos_produtos = []
    pagina_atual = 1
    
    while len(todos_produtos) < max_produtos:
        print(f"\n--- Processando Página {pagina_atual} ---")
        page.scroll.to_bottom()
        time.sleep(3)
        
        markdown_vitrine = html_para_markdown(page.html)
        resultado_stage1 = extrair_dados_estagio_1(markdown_vitrine, dominio, termo_busca)
        produtos_pagina = resultado_stage1.get('products', [])
        
        if not produtos_pagina: break
        todos_produtos.extend(produtos_pagina)
        print(f"✅ Acumulado: {len(todos_produtos)} produtos.")

        if len(todos_produtos) >= max_produtos:
            todos_produtos = todos_produtos[:max_produtos]
            break
        
        botoes_prox = ['text:Próxima', 'text:Next', 'text:>', '@class:next', '@aria-label:Next', '.shopee-icon-button--right']
        proxima_pagina_url = None
        botao_encontrado = None
        
        for seletor in botoes_prox:
            btn = page.ele(seletor, timeout=2)
            if btn:
                botao_encontrado = btn
                try:
                    href = btn.attr('href')
                    if href and href.strip() not in ['', '#', 'javascript:void(0);']:
                        proxima_pagina_url = f"https://{dominio}/{href.lstrip('/')}" if not href.startswith('http') else href
                        break
                except: pass
                
        if proxima_pagina_url:
            page.get(proxima_pagina_url)
            time.sleep(5)
        elif botao_encontrado:
            try:
                botao_encontrado.scroll.to_see()
                page.run_js("arguments.click();", botao_encontrado)
                time.sleep(6)
            except: break
        else: break
        pagina_atual += 1

    # Filtra links baseado na flag de verificação
    links_finais = [p for p in todos_produtos if p.get('link') not in cache_verificacao]
    print(f"\n✅ ESTÁGIO 1 CONCLUÍDO: {len(links_finais)} novos links para processar.")

    # Definição das pastas e nomes de arquivos de saída para salvamento incremental
    os.makedirs("resultados/todos_os_itens", exist_ok=True)
    os.makedirs("resultados/itens_encontrados", exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dominio_slug = dominio.replace('.', '_')
    
    # Caminhos para "Todos os itens"
    arquivo_json_all = f"resultados/todos_os_itens/pipeline_{dominio_slug}_{timestamp}.json"
    arquivo_csv_all = f"resultados/todos_os_itens/pipeline_{dominio_slug}_{timestamp}.csv"
    
    # Caminhos para "Itens encontrados"
    arquivo_json_found = f"resultados/itens_encontrados/pipeline_{dominio_slug}_{timestamp}.json"
    arquivo_csv_found = f"resultados/itens_encontrados/pipeline_{dominio_slug}_{timestamp}.csv"

    print("\n🚀 INICIANDO ESTÁGIO 2: EXTRAÇÃO PROFUNDA")
    
    for i, produto in enumerate(links_finais):
        link_produto = produto.get('link')
        if not link_produto.startswith('http'):
            link_produto = f"https://{dominio}/{link_produto.lstrip('/')}"
            
        try:
            page.get(link_produto)
            time.sleep(4)
        except Exception as e:
            print(f"   ⚠️ Erro ao carregar página {i+1}: {e}")
            continue
        
        def explorar_e_extrair():
            html_otimizado = limpar_html_agnostico(page.html)
            res, prompt = extrair_dados_estagio_2(html_otimizado, info_alvo)
            dados = {item['chave']: item['valor'] for item in res.get('detalhes', [])}
            return dados, res, prompt

        seletor_vencedor = CLICK_MEMORY.get(dominio)
        if seletor_vencedor:
            try: page.ele(seletor_vencedor, timeout=2).click(by_js=True); time.sleep(2)
            except: pass

        dados_extras, resultado_stage2, prompt_utilizado = explorar_e_extrair()
        encontrou_info = verificar_se_achou(dados_extras)

        if not encontrou_info:
            max_tentativas = 3
            tentativa = 0
            alvos = ['@role=tab', 'text:Detalhes', 'text:Especificações', 'text:Informações gerais', 'text:Ver mais', 'text:Ver tudo']

            while not encontrou_info and tentativa < max_tentativas:
                interativos = []
                for a in alvos:
                    try: interativos.extend(page.eles(a))
                    except: continue

                for el in interativos:
                    try:
                        texto_el = el.text.lower()
                        if any(x in texto_el for x in ['avaliação', 'comentário', 'review']):
                            continue
                        
                        seletor_atual = f"text:{el.text}" if el.text else f"xpath:{el.xpath}"
                        el.scroll.to_see()
                        el.click(by_js=True)
                        time.sleep(2)
                        
                        dados_extras, resultado_stage2, prompt_utilizado = explorar_e_extrair()
                        if verificar_se_achou(dados_extras):
                            encontrou_info = True
                            CLICK_MEMORY[dominio] = seletor_atual
                            break
                    except:
                        continue
                tentativa += 1

        content_markdown = html_para_markdown(page.html)
        registro_completo = {
            "name": produto.get('name'), "link": link_produto,
            "schema": ProductDetails.model_json_schema(), "content": content_markdown,
            "prompt": prompt_utilizado, "response": json.dumps(resultado_stage2, ensure_ascii=False),
            **dados_extras
        }
        
        # Salva sempre no cache geral de processados
        salvar_no_cache(link_produto, tipo="all")

        # --- LÓGICA DE PERSISTÊNCIA INCREMENTAL ---
        keys_to_save = ["name", "link"] + list(dados_extras.keys())
        item_saida = {k: registro_completo.get(k, "Não encontrado") for k in keys_to_save}

        # 1. Salva sempre em "Todos os itens"
        # JSON All
        data_json_all = []
        if os.path.exists(arquivo_json_all):
            with open(arquivo_json_all, 'r', encoding='utf-8') as fj:
                try: data_json_all = json.load(fj)
                except: data_json_all = []
        data_json_all.append(item_saida)
        with open(arquivo_json_all, 'w', encoding='utf-8') as fj:
            json.dump(data_json_all, fj, indent=4, ensure_ascii=False)

        # CSV All
        file_exists_all = os.path.exists(arquivo_csv_all)
        with open(arquivo_csv_all, 'a', encoding='utf-8', newline='') as fc:
            writer = csv.DictWriter(fc, fieldnames=keys_to_save)
            if not file_exists_all: writer.writeheader()
            writer.writerow(item_saida)

        # 2. Salva em "Itens encontrados" apenas se encontrou info ou flag ativa
        if encontrou_info or SALVAR_NAO_ENCONTRADOS == 1:
            salvar_checkpoint(registro_completo)
            salvar_no_cache(link_produto, tipo="found")
            
            # JSON Found
            data_json_found = []
            if os.path.exists(arquivo_json_found):
                with open(arquivo_json_found, 'r', encoding='utf-8') as fj:
                    try: data_json_found = json.load(fj)
                    except: data_json_found = []
            data_json_found.append(item_saida)
            with open(arquivo_json_found, 'w', encoding='utf-8') as fj:
                json.dump(data_json_found, fj, indent=4, ensure_ascii=False)

            # CSV Found
            file_exists_found = os.path.exists(arquivo_csv_found)
            with open(arquivo_csv_found, 'a', encoding='utf-8', newline='') as fc:
                writer = csv.DictWriter(fc, fieldnames=keys_to_save)
                if not file_exists_found: writer.writeheader()
                writer.writerow(item_saida)

            print(f"[{i+1}/{len(links_finais)}] ✅ Finalizado e Persistido.")
        else:
            print(f"[{i+1}/{len(links_finais)}] ⚠️ Mal sucedido.")

    page.quit()
    print(f"\n💾 Processo concluído. Arquivos organizados na pasta 'resultados/'.")

if __name__ == "__main__":
    main()