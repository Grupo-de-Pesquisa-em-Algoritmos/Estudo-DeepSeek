from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer, TextIteratorStreamer
import torch
from threading import Thread
import sys
model_name = "deepseek-ai/deepseek-llm-7b-chat"
print("Carregando o modelo e o tokenizador '{model_name}'...")
model = None;
try :
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
except:
    print("\033[31;1mbits and bytes indisponível\033[0m",file=sys.stderr);
    model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

if torch.cuda.is_available():
    model.to("cuda")
    print("Modelo movido para a GPU.")
else:
    print("\033[31;1mGPU NVidia não disponível. O modelo será executado na CPU.\033[0m",file=sys.stderr)

#prompt = "ola tudo bem qual o seu nome escreva paragrafo sobre impactos IA meio ambiente."

prompt = "oi, escreva um paragrafo sobre o uso de ias generativas."
print("\nPrompt de entrada:")
print(prompt)

messages = [
    {"role": "system", "content": "[[always answer in brazilian portuguese]]"},
    {"role": "system", "content": "[ai called TESTE is concise]]"},
    {"role": "user", "content": prompt}
]

input = tokenizer.apply_chat_template(messages, return_tensors="pt")
input_ids = None;
if torch.cuda.is_available():
    input_ids = input.to("cuda")
else:
  input_ids = input["input_ids"];
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
new_thread = Thread(target = model.generate, kwargs={
    "input_ids" :input_ids,
    "max_new_tokens" : 500,
    "do_sample" : True,
    "attention_mask" : input["attention_mask"],
    "temperature" : 0.7,
    "eos_token_id" : model.config.eos_token_id,
    "pad_token_id" : model.config.eos_token_id,
#    "num_return_sequences" : 1,
    "repetition_penalty" : 1.0,
    "streamer" : streamer                                              
    })
new_thread.start()
for tok in streamer:
    print(tok, end="")
    sys.stdout.flush()
print("\nGENERATION DONE\n");
#text_streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True);
#output = model.generate(
#    input_ids,
#    max_new_tokens=50,
#    do_sample=True,
#    temperature=0.001,
#    eos_token_id=model.config.eos_token_id,
#    pad_token_id=model.config.eos_token_id,
#    streamer=text_streamer
#)
#end = datetime.now()!time_difference = (end- start).total_seconds() * 10**3!response = tokenizer.decode(output[0], skip_special_tokens=True)!print(time_difference, "ms generation\n");!print("\n--- Resposta ---")!print(response)!print("----------------")!
