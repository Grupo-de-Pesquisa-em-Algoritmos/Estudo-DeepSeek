import time
import torch
import psutil
import os
import gc
import shutil
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- CONFIGURAÇÕES ---
PROMPT_ENTRADA = "Olá! Qual o seu nome e quem te criou?"
MAX_CARACTERES = 1000
DATA_ATUAL = datetime.now().strftime("%Y-%m-%d")
NOME_ARQUIVO_SAIDA = f"testes_IAs_{DATA_ATUAL}.txt"

# Caminho para a pasta de SWAP (no mesmo diretório do script)
DIRETORIO_SCRIPT = os.path.dirname(os.path.abspath(__file__))
PASTA_SWAP = os.path.join(DIRETORIO_SCRIPT, "swap")

MODELOS_PARA_TESTAR = [
    {"nome": "DeepSeek-Coder-1.3B", "id": "deepseek-ai/deepseek-coder-1.3b-instruct"},
    {"nome": "Llama-3.2-3B", "id": "meta-llama/Llama-3.2-3B-Instruct"},
    {"nome": "Gemma-2-2B (Google)", "id": "google/gemma-2-2b-it"},
    {"nome": "Qwen2.5-3B", "id": "Qwen/Qwen2.5-3B-Instruct"},
    {"nome": "DeepSeek-LLM-7B", "id": "deepseek-ai/deepseek-llm-7b-chat"}
]

def limpar_tela_visual():
    os.system('cls' if os.name == 'nt' else 'clear')

def obter_uso_ram():
    processo = psutil.Process(os.getpid())
    uso_bytes = processo.memory_info().rss
    return f"{uso_bytes / (1024 ** 3):.2f} GB"

def limpar_memoria_gpu():
    """Limpeza profunda de memória."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def limpar_pasta_swap():
    """Remove a pasta de swap se ela existir para liberar espaço em disco."""
    if os.path.exists(PASTA_SWAP):
        try:
            shutil.rmtree(PASTA_SWAP)
        except Exception as e:
            print(f"Aviso: Não foi possível apagar a pasta swap: {e}")

def log_sistema(mensagem):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensagem}")

def salvar_resultado(texto):
    with open(NOME_ARQUIVO_SAIDA, "a", encoding="utf-8") as f:
        f.write(texto + "\n" + "="*50 + "\n")

def rodar_modelo(modelo_info):
    nome_amigavel = modelo_info["nome"]
    modelo_id = modelo_info["id"]
    
    limpar_tela_visual()
    
    print("="*50)
    log_sistema(f"PREPARANDO: {nome_amigavel}")
    log_sistema(f"ID HuggingFace: {modelo_id}")
    log_sistema(f"Pasta de Swap configurada: {PASTA_SWAP}")
    print("="*50)
    
    try:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16 if device == "cuda" else torch.float32

        log_sistema("Carregando Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(modelo_id, trust_remote_code=True)
        
        log_sistema("Carregando Modelo (Se faltar RAM, usará o DISCO)...")
        
        # Garante que a pasta swap existe antes de começar
        os.makedirs(PASTA_SWAP, exist_ok=True)

        modelo = AutoModelForCausalLM.from_pretrained(
            modelo_id,
            device_map="auto", 
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            offload_folder=PASTA_SWAP,  # <--- Define a pasta local ./swap
            offload_state_dict=True     # <--- Permite jogar pesos para o disco
        )

        ram_inicial = obter_uso_ram()
        
        messages = [{"role": "user", "content": PROMPT_ENTRADA}]
        
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(modelo.device)

        log_sistema("Gerando resposta...")
        start_time = time.time()

        with torch.no_grad():
            outputs = modelo.generate(
                inputs, 
                max_new_tokens=500, 
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id
            )
        
        end_time = time.time()
        tempo_total = end_time - start_time
        
        tokens_gerados = outputs[0][len(inputs[0]):]
        resposta_completa = tokenizer.decode(tokens_gerados, skip_special_tokens=True)
        
        if len(resposta_completa) > MAX_CARACTERES:
            resposta_completa = resposta_completa[:MAX_CARACTERES] + "... [CORTADO]"

        ram_final = obter_uso_ram()
        
        log_texto = (
            f"MODELO: {nome_amigavel} ({modelo_id})\n"
            f"TEMPO DE EXECUÇÃO: {tempo_total:.4f} segundos\n"
            f"RAM NO INÍCIO: {ram_inicial} | RAM NO FINAL: {ram_final}\n"
            f"PROMPT: {PROMPT_ENTRADA}\n"
            f"RESPOSTA:\n{resposta_completa}\n"
        )
        
        salvar_resultado(log_texto)
        log_sistema(f"Sucesso! Tempo: {tempo_total:.2f}s.")
        print(f"\nPrévia da resposta:\n{resposta_completa[:200]}...\n")

        # Limpeza
        del inputs, outputs, modelo, tokenizer, tokens_gerados
        limpar_memoria_gpu()
        limpar_pasta_swap() # Limpa o disco

    except Exception as e:
        erro_msg = f"ERRO CRÍTICO em {nome_amigavel}: {str(e)}"
        print(f"\n!!! {erro_msg} !!!\n")
        salvar_resultado(f"MODELO: {nome_amigavel}\nSTATUS: FALHA\nERRO: {str(e)}\n")
        limpar_memoria_gpu()
        limpar_pasta_swap()

    time.sleep(2) 

if __name__ == "__main__":
    limpar_tela_visual()
    
    with open(NOME_ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        f.write(f"RELATÓRIO DE TESTE DE IAs - {DATA_ATUAL}\n")
        f.write(f"Hardware GPU: {'Sim' if torch.cuda.is_available() else 'Não'}\n")
        f.write("="*50 + "\n")

    print(f"Iniciando bateria de testes com {len(MODELOS_PARA_TESTAR)} modelos...\n")
    print(f"AVISO: O modelo DeepSeek-LLM-7B pode usar o disco (swap) se a RAM encher.\nIsso criará arquivos temporários em: {PASTA_SWAP}\n")
    time.sleep(3)
    
    for modelo in MODELOS_PARA_TESTAR:
        rodar_modelo(modelo)
    
    print("\nBateria de testes finalizada! Verifique o arquivo .txt")
