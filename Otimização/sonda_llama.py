import torch
import pickle
import math
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm

# --- CONFIGURAÇÕES ---
MODEL_ID = "unsloth/Llama-3.1-8B-Instruct" # Versão otimizada
INPUT_FILE = "words.txt"
OUTPUT_FILE = "mapa_neuronios_ptbr.pkl"
BATCH_SIZE = 64  # Perfeito para a L4 (24GB)

def main():
    print(f"--- INICIANDO SONDA DE NEURÔNIOS (Llama 3.1 8B) ---")
    
    # 1. Carregar Modelo em 4-bit (NF4)
    print(">>> Carregando modelo...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token # Necessário para batching
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        attn_implementation="sdpa"
    )

    # 2. Configurar os Ganchos (Hooks)
    # Llama 3 8B: 32 camadas, 14336 neurônios intermediários
    n_layers = model.config.num_hidden_layers
    mlp_size = model.config.intermediate_size
    
    # Tensor para contar ativações (Na GPU para velocidade)
    neuron_activity = torch.zeros((n_layers, mlp_size), dtype=torch.int32, device="cuda")
    print(f">>> Monitorando {n_layers} camadas x {mlp_size} neurônios.")

    def get_activation_hook(layer_idx):
        def hook(module, input, output):
            # input[0] é o tensor de entrada da camada down_proj
            # Shape: [Batch, Seq_Len, Neurons]
            # Queremos saber quais neurônios "acenderam" (magnitude > 0)
            
            activation_tensor = input[0]
            
            # Lógica: Se o valor absoluto da ativação for maior que um limiar mínimo, conta como uso.
            # .any(dim=1) -> Se ativou em qualquer token da frase
            # .sum(dim=0) -> Soma quantas frases do batch ativaram esse neurônio
            hits = (activation_tensor.abs() > 1e-4).any(dim=1).int().sum(dim=0)
            
            neuron_activity[layer_idx] += hits
        return hook

    # Registrar hooks na camada 'down_proj' (onde os neurônios convergem)
    for i, layer in enumerate(model.model.layers):
        layer.mlp.down_proj.register_forward_hook(get_activation_hook(i))

    # 3. Carregar Palavras
    print(f">>> Lendo {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            palavras = [line.strip() for line in f.readlines() if line.strip()]
        print(f"Total de palavras: {len(palavras)}")
    except FileNotFoundError:
        print("ERRO: Arquivo words.txt não encontrado.")
        return

    # 4. Processamento em Lote
    num_batches = math.ceil(len(palavras) / BATCH_SIZE)
    
    print(f">>> Processando em {num_batches} lotes...")
    
    for i in tqdm(range(0, len(palavras), BATCH_SIZE)):
        batch_words = palavras[i : i + BATCH_SIZE]
        
        # O prompt força o modelo a buscar o significado da palavra
        prompts = [f"O significado da palavra '{p}' é" for p in batch_words]
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=32 # Curto para ser rápido
        ).to("cuda")
        
        with torch.no_grad():
            # Apenas o forward pass já ativa os hooks.
            # Não precisamos de model.generate() (que é lento).
            model(**inputs)

    # 5. Salvar Resultados
    print(f">>> Salvando mapa em {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump(neuron_activity.cpu(), f)
        
    print(">>> Concluído! Agora você tem o mapa do tesouro.")

if __name__ == "__main__":
    main()
