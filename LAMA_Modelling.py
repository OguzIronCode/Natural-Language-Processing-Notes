from transformers import AutoTokenizer, AutoModelForCausalLM

# Model ve Tokenizer Yükle

model_name="huggyLLama/LLama-7b"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModelForCausalLM.from_pretrained(model_name)

# Ornek Metin

text="ı go to swim for"

# Tokenize

inputs=tokenizer(text,return_tensors="pt")

# Metin Tamamlama

output=model.generate(inputs.input_ids,max_length=10)

# Decode

generated_text=tokenizer.decode(output[0],skip_special_tokens=True)
print(generated_text)