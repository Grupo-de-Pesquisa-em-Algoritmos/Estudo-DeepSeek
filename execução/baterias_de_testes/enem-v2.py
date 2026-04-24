import time
import psutil
from llama_cpp import Llama

def obter_metricas_sistema():
    mem = psutil.virtual_memory()
    ram_usada = mem.used / (1024**3)
    return ram_usada

def executar_teste():
    modelos = ["modelos/deepseek-llm-7b-chat.Q6_K.gguf"]
    prompts = [
        "Invisibilidade e registro civil: garantia de acesso à cidadania no Brasil",
        "Desafios para a valorização de comunidades e povos tradicionais no Brasil",
        "Desafios para o enfrentamento da invisibilidade do trabalho de cuidado realizado pela mulher no Brasil",
        "Desafios para a valorização da herança africana no Brasil",
        "Perspectivas acerca do envelhecimento na sociedade brasileira"
    ]
    
    for modelo in modelos:
        llm = Llama(
            model_path=modelo,
            n_gpu_layers=0, 
            n_ctx=256,
            verbose=True
        )

        for prompt in prompts:

            system_prompt = "Redija uma redação modelo Dissertativa Argumentativa com proposta de intervenção seguindo o modelo ENEM brasileiro. Sua resposta deve ser apenas a redação."
            prompt_completo = f"{system_prompt}\n\nUsuário: {prompt}\n\nAssistente:\n"

            inicio_tempo = time.time()

            resposta = llm(prompt_completo, max_tokens=4000)
            
            tempo_total = time.time() - inicio_tempo
            tokens_gerados = resposta['usage']['completion_tokens']
            tps = tokens_gerados / tempo_total
            texto_resposta = resposta['choices'][0]['text']

            ram_usada = obter_metricas_sistema()

            conteudo_arquivo = (
                f"=== Relatório de Performance ({modelo}) ===\n"
                f"Tempo de execução: {tempo_total:.2f} s\n"
                f"Tokens por segundo: {tps:.2f} t/s\n"
                f"RAM Total Utilizada (Unificada): {ram_usada:.2f} GB\n\n"
                "=== Resposta do Modelo ===\n"
                f"{texto_resposta}\n\n"
            )

            with open("resultados_enem-v2.txt", "a", encoding="utf-8") as f:
                f.write(conteudo_arquivo)
                
        del llm

if __name__ == "__main__":
    executar_teste()
