from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_name = "deepseek-ai/deepseek-llm-7b-chat"
print(f"Carregando o modelo e o tokenizador '{model_name}'...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if torch.cuda.is_available():
    model.to("cuda")
    print("Modelo movido para a GPU.")
else:
    print("GPU NVidia não disponível. O modelo será executado na CPU.")

prompt = "Olá! Tudo bem? Qual o seu nome?"

print("\nPrompt de entrada:")

print(prompt)

messages = [
    {"role": "user", "content": prompt}
]

input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt")

if torch.cuda.is_available():
    input_ids = input_ids.to("cuda")

print("\nGerando texto...")
output = model.generate(
    input_ids,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.7,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n--- Resposta ---")
print(response)
print("----------------")
