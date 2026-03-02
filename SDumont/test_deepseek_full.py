import time
import pynvml
import os
from vllm import LLM, SamplingParams

pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
gpu_name = pynvml.nvmlDeviceGetName(handle)
cpu_info = os.popen("lscpu | grep -E 'Model name|MHz|CPU\(s\):'").read().strip()

prompts = [
    "O que é supercomputação?", "Escreva um poema sobre IA.",
    "Como inverter uma lista em Python?", "Explique a teoria da relatividade.",
    "Qual o sentido da vida?", "Traduza 'bom dia' para 5 idiomas.",
    "Resuma a história do Brasil em 1 parágrafo.", "Crie uma piada sobre programadores.",
    "O que é computação quântica?", "Como funciona um transformer?"
]

# Configuração para 20 GPUs (ex: 5 nós x 4 GPUs)
llm = LLM(
    model="deepseek-ai/DeepSeek-R1",
    tensor_parallel_size=4,
    pipeline_parallel_size=5,
    trust_remote_code=True,
    dtype="bfloat16"
)
sampling_params = SamplingParams(temperature=0.7, max_tokens=256)

with open("resultados_deepseek_full.txt", "w", encoding="utf-8") as f:
    f.write(f"--- Hardware (Nó Principal) ---\nGPU: {gpu_name}\nCPU:\n{cpu_info}\n\n")

    for i, p in enumerate(prompts):
        t0 = time.time()
        output = llm.generate([p], sampling_params, use_tqdm=False)[0]
        t_total = time.time() - t0
        
        if hasattr(output, 'metrics') and output.metrics is not None and output.metrics.first_token_time:
            ttft = output.metrics.first_token_time - output.metrics.first_scheduled_time
        else:
            ttft = 0.0
            
        texto = output.outputs[0].text
        watts = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
        vram_usada = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024**3)

        f.write(f"--- Prompt {i+1} ---\n{p}\nTexto: {texto}\n")
        f.write(f"VRAM (Principal): {vram_usada:.2f} GB | TTFT: {ttft:.3f} s | Tempo Total: {t_total:.2f} s | Potência: {watts:.2f} W\n\n")

pynvml.nvmlShutdown()
