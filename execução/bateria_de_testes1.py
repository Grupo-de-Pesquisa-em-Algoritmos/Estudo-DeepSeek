import time
import torch
import psutil
import os
import gc
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPT_ENTRADA = "Olá! Qual o seu nome?"
MAX_CARACTERES = 1000
DATA_ATUAL = datetime.now().strftime("%Y-%m-%d")
NOME_ARQUIVO_SAIDA = f"testes_IAs_{DATA_ATUAL}.txt"

MODELOS_PARA_TESTAR = [
    {"nome": "DeepSeek-Coder-1.3B", "id": "deepseek-ai/deepseek-coder-1.3b-instruct"},
    {"nome": "Llama-3.2-3B", "id": "meta-llama/Llama-3.2-3B-Instruct"},
    {"nome": "Gemma-2-2B (Google)", "id": "google/gemma-2-2b-it"},
    {"nome": "Qwen2.5-3B", "id": "Qwen/Qwen2.5-3B-Instruct"},
    {"nome": "DeepSeek-LLM-7B", "id": "deepseek-ai/deepseek-llm-7b-chat"}
]

def obter_uso_ram():
    processo = psutil.Process(os.getpid())
    uso_bytes = processo.memory_info().rss
    return f"{uso_bytes / (1024 ** 3):.2f} GB"

def limpar_memoria():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def log_sistema(mensagem):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {mensagem}")

def salvar_resultado(texto):
    with open(NOME_ARQUIVO_SAIDA, "a", encoding="utf-8") as f:
        f.write(texto + "\n" + "="*50 + "\n")

def rodar_modelo(modelo_info):
    nome_amigavel = modelo_info["nome"]
    modelo_id = modelo_info["id"]
    
    log_sistema(f"--- Iniciando teste: {nome_amigavel} ---")
    log_sistema(f"Baixando/Carregando modelo: {modelo_id}...")
    
    try:
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        tokenizer = AutoTokenizer.from_pretrained(modelo_id, trust_remote_code=True)
        
        modelo = AutoModelForCausalLM.from_pretrained(
            modelo_id,
            device_map="auto", 
            torch_dtype=dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        ram_inicial = obter_uso_ram()
        log_sistema(f"Modelo carregado. RAM usada pelo script: {ram_inicial}. Dispositivo: {modelo.device}")

        inputs = tokenizer(PROMPT_ENTRADA, return_tensors="pt").to(modelo.device)

        log_sistema("Gerando resposta...")
        start_time = time.time()

        with torch.no_grad():
            outputs = modelo.generate(
                **inputs, 
                max_new_tokens=500, 
                do_sample=True,
                temperature=0.7
            )
        
        end_time = time.time()
        tempo_total = end_time - start_time
        
        resposta_completa = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
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
        log_sistema(f"Sucesso! Tempo: {tempo_total:.2f}s. Salvo em {NOME_ARQUIVO_SAIDA}")

        del inputs, outputs, modelo, tokenizer
        limpar_memoria()

    except Exception as e:
        erro_msg = f"ERRO ao rodar {nome_amigavel}: {str(e)}"
        log_sistema(erro_msg)
        salvar_resultado(f"MODELO: {nome_amigavel}\nSTATUS: FALHA\nERRO: {str(e)}\n")
        limpar_memoria()

if __name__ == "__main__":
    with open(NOME_ARQUIVO_SAIDA, "w", encoding="utf-8") as f:
        f.write(f"RELATÓRIO DE TESTE DE IAs - {DATA_ATUAL}\n")
        f.write(f"Hardware Detectado: GPU={'Sim' if torch.cuda.is_available() else 'Não'}\n")
        f.write("="*50 + "\n")

    print(f"Iniciando bateria de testes com {len(MODELOS_PARA_TESTAR)} modelos...\n")
    
    for modelo in MODELOS_PARA_TESTAR:
        rodar_modelo(modelo)
        print("-" * 30)
    
    print("\nBateria de testes finalizada!")
