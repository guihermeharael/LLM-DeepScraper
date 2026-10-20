# ==============================================================================
# 1. INSTALAÇÃO DE DEPENDÊNCIAS (Executar numa célula de notebook ou terminal)
# ==============================================================================
"""
%%capture
import os, re
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    import torch; v = re.match(r'[\d]{1,}\.[\d]{1,}', str(torch.__version__)).group(0)
    xformers = 'xformers==' + {'2.10':'0.0.34','2.9':'0.0.33.post1','2.8':'0.0.32.post2'}.get(v, "0.0.34")
    !pip install sentencepiece protobuf "datasets==4.3.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth_zoo bitsandbytes accelerate {xformers} peft trl triton unsloth
    !pip install --no-deps --upgrade "torchao>=0.16.0"
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
"""

# ==============================================================================
# 2. CARREGAMENTO DO MODELO BASE E TOKENIZER
# ==============================================================================
from unsloth import FastLanguageModel
import torch

max_seq_length = 2048 # Suporta RoPE Scaling internamente
dtype = None # None para deteção automática
load_in_4bit = True # Quantização 4bit para reduzir uso de memória

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.1-8B",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# Adicionar adaptadores LoRA
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# ==============================================================================
# 3. PREPARAÇÃO DOS DADOS E PREVENÇÃO DE DATA LEAKAGE (HASH SPLIT)
# ==============================================================================
import json
import hashlib
import random
from collections import defaultdict

# Agrupar os itens pelo Hash do conteúdo para evitar vazamento
data_by_hash = defaultdict(list)
with open("dataset_final.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        item = json.loads(line)
        conteudo_base = item["content"]
        page_hash = hashlib.md5(conteudo_base.encode('utf-8')).hexdigest()
        data_by_hash[page_hash].append(item)

# Separar 80% dos hashes para treino e 20% para validação
hashes = list(data_by_hash.keys())
random.seed(42)
random.shuffle(hashes)

split_idx = int(len(hashes) * 0.8)
train_hashes = set(hashes[:split_idx])

train_data, val_data = [], []
for h, items in data_by_hash.items():
    if h in train_hashes:
        train_data.extend(items)
    else:
        val_data.extend(items)

# Salvar os novos ficheiros isolados
with open("train_hash.jsonl", "w", encoding="utf-8") as f:
    for d in train_data: f.write(json.dumps(d) + "\n")
with open("val_hash.jsonl", "w", encoding="utf-8") as f:
    for d in val_data: f.write(json.dumps(d) + "\n")

print(f"Treino: {len(train_data)} exemplos | Validação: {len(val_data)} exemplos")

# ==============================================================================
# 4. FORMATAÇÃO E MAPEAMENTO DO DATASET
# ==============================================================================
alpaca_prompt = """Abaixo está uma instrução que descreve uma tarefa, acompanhada de uma entrada que fornece mais contexto. Escreva uma resposta que complete adequadamente o pedido.\n\n### Instrução:\n{}\n\n### Entrada (Conteúdo):\n{}\n\n### Resposta:\n{}"""

EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    prompts   = examples["prompt"]
    contents  = examples["content"]
    responses = examples["response"]

    texts = []
    for prompt, content, response in zip(prompts, contents, responses):
        text = alpaca_prompt.format(prompt, content, response) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts }

from datasets import load_dataset
raw_datasets = load_dataset("json", data_files={"train": "train_hash.jsonl", "validation": "val_hash.jsonl"})

train_dataset = raw_datasets["train"].map(formatting_prompts_func, batched = True)
val_dataset = raw_datasets["validation"].map(formatting_prompts_func, batched = True)

# ==============================================================================
# 5. TREINO DO MODELO E ESTATÍSTICAS
# ==============================================================================
from trl import SFTConfig, SFTTrainer

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = val_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    packing = False,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1,
        learning_rate = 2e-4,
        logging_steps = 1,
        eval_strategy = "steps",
        eval_steps = 25,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", 
    ),
)

# Estatísticas iniciais de memória
gpu_stats = torch.cuda.get_device_properties(0)
start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
print(f"{start_gpu_memory} GB of memory reserved.")

# Iniciar o treino
trainer_stats = trainer.train()

# Estatísticas finais de memória e tempo
used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
used_percentage = round(used_memory / max_memory * 100, 3)
lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
print(f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training.")
print(f"Peak reserved memory = {used_memory} GB.")
print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
print(f"Peak reserved memory % of max memory = {used_percentage} %.")
print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

# ==============================================================================
# 6. INFERÊNCIA E TESTES
# ==============================================================================
FastLanguageModel.for_inference(model)

inputs = tokenizer(
[
    alpaca_prompt.format(
        "Continue the fibonacci sequence.", # instruction
        "1, 1, 2, 3, 5, 8", # input
        "", # output (deixar em branco para geração)
    )
], return_tensors = "pt").to("cuda")

# Inferência com geração completa
outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
print(tokenizer.batch_decode(outputs))

# Inferência com TextStreamer (visualização token a token)
from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 128)

# ==============================================================================
# 7. GUARDAR O MODELO (LORA E MERGED) E EXPORTAÇÃO GGUF PARA O HUGGING FACE
# ==============================================================================
# Guardar apenas os adaptadores LoRA localmente
model.save_pretrained("llama_lora")  
tokenizer.save_pretrained("llama_lora")

# Guardar o modelo em formato merged de 16bit localmente
model.save_pretrained_merged("llama_merged", tokenizer, save_method="merged_16bit")

# Limpeza de cache do llama.cpp antes da conversão
import shutil, os
llama_path = os.path.expanduser("~/.unsloth/llama.cpp")
if os.path.exists(llama_path):
    shutil.rmtree(llama_path)
    print("Cache limpo!")

# EXPORTAÇÃO PARA O HUGGING FACE 
# Altere as variáveis abaixo para "if True:" quando desejar executar as respetivas funções de upload

# Exportar para 8bit Q8_0
if False: model.save_pretrained_gguf("llama_finetune", tokenizer)

# Lembrete: aceda a https://huggingface.co/settings/tokens para gerar o seu token!
# Altere "SEU_USUARIO_HF" e "SEU_TOKEN_HF_AQUI" adequadamente.
if False: model.push_to_hub_gguf(
    "SEU_USUARIO_HF/NOME_DO_SEU_MODELO_Q8_0",
    tokenizer,
    quantization_method = "q8_0",
    token = "SEU_TOKEN_HF_AQUI"
)

# Exportar para 16bit f16
if False: model.save_pretrained_gguf("llama_finetune", tokenizer, quantization_method = "f16")
if False: model.push_to_hub_gguf("SEU_USUARIO_HF/NOME_DO_SEU_MODELO_f16", tokenizer, quantization_method = "f16", token = "SEU_TOKEN_HF_AQUI")

# Exportar para 4bit q4_k_m
if False: model.save_pretrained_gguf("llama_finetune", tokenizer, quantization_method = "q4_k_m")
if False: model.push_to_hub_gguf(
    "SEU_USUARIO_HF/NOME_DO_SEU_MODELO_q4_k_m",
    tokenizer,
    quantization_method = "q4_k_m",
    token = "SEU_TOKEN_HF_AQUI"
)

# Exportar para múltiplos métodos GGUF num único ciclo
if False:
    for method in ["q4_k_m", "q8_0", "q5_k_m"]:
        model.push_to_hub_gguf(
            "SEU_USUARIO_HF/NOME_DO_SEU_MODELO", 
            tokenizer,
            quantization_method = method,
            token = "SEU_TOKEN_HF_AQUI",
        )