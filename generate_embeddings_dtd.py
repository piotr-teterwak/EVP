import numpy as np
import torch
import torchvision.datasets as datasets
from transformers import AutoTokenizer, AutoConfig
from auto_gptq import AutoGPTQForCausalLM
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModelForCausalLM
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from huggingface_hub import snapshot_download




# 1. Load in Vicuna 30B
MODEL = "/fsx/pteterwak/llama_cache/wizard_models/Wizard-Vicuna-30B-Uncensored-GPTQ"
model_basename ="Wizard-Vicuna-30B-Uncensored-GPTQ-4bit.act.order"




tokenizer = AutoTokenizer.from_pretrained(MODEL, use_fast=True)
model = AutoGPTQForCausalLM.from_quantized(MODEL,
        model_basename=model_basename,
        use_safetensors=True,
        use_triton=False,
        device_map="auto",
        quantize_config=None)


#tokenizer = LlamaTokenizer.from_pretrained("eachadea/vicuna-13b-1.1")
##model = LlamaForCausalLM.from_pretrained("eachadea/vicuna-13b-1.1", device_map="auto")
#
#checkpoint = "eachadea/vicuna-13b-1.1"
#weights_location = snapshot_download(checkpoint)
#
#
##Create a model and initialize it with empty weights
#config = AutoConfig.from_pretrained(checkpoint)
#with init_empty_weights():
#    model = AutoModelForCausalLM.from_config(config)
#
##model = model.tie_weights()
#
##Load the checkpoint and dispatch it to the right devices
#model = load_checkpoint_and_dispatch(
#    model, weights_location, device_map="auto", no_split_module_classes=["LlamaDecoderLayer"]
#)


#2. Get list of classes for DTD


# Load DTD
train_dataset = datasets.DTD(root='/fsx/pteterwak/data/dtd')

# Retrieve list of classes
classes = train_dataset.classes

# Print the list of classes
for i, cls in enumerate(classes):
    print(f'Class {i}: {cls}')


#3. Generate and print prompts. 

prompt_template = "### Instruction: The dataset is describable textures. The class name is \"{}\".How would you describe {} in terms of simple shapes, colors, and textures? Be detailed but succinct.  ### Assistant: "
#prompt_template = "The dataset is CIFAR-100. The class name is \"{}\".How would you describe {} in terms of simple shapes, colors, and textures? Be detailed but succinct."

representation_list = []
text_list = []

for cls in classes:
    prompt = prompt_template.format(cls, cls)

    batch = tokenizer(prompt, return_tensors = "pt")["input_ids"].cuda()
    outputs = model.generate(input_ids=batch, max_new_tokens=400, min_new_tokens=20,num_beams=5,return_dict_in_generate=True, output_hidden_states =True)
    text = tokenizer.batch_decode(outputs['sequences'], skip_special_tokens=True)
    print(text)
    text_list.append(text)

    hidden_states_list = [[i[0][0].cpu() for i in j] for j in outputs['hidden_states']]
    hidden_states_list_1 = [torch.stack(t) for t in hidden_states_list]

    hidden_states_np = torch.stack(hidden_states_list_1).numpy()[:,-1,:]
    representation_list.append(hidden_states_np)

np.savez('embeddings/representations_dtd.npz', *representation_list)
np.savez('embeddings/text_dtd.npz', *text_list)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
