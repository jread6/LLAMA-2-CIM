import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer
import transformers

model = LlamaForCausalLM.from_pretrained('/usr/scratch/llama_model/models--meta-llama--Llama-2-13b-chat-hf/snapshots/13f8d72c0456c17e41b3d8b4327259125cd0defa')
tokenizer = LlamaTokenizer.from_pretrained('/usr/scratch/llama_model/models--meta-llama--Llama-2-13b-chat-hf/snapshots/13f8d72c0456c17e41b3d8b4327259125cd0defa')
 
# text = ' '.join(['hello']*4095)
# with torch.no_grad():
#     input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)
#     outputs = model(input_ids)
#     print(tokenizer.decode(outputs))

# tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
    tokenizer=tokenizer,
)

sequences = pipeline(
    'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n',
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")