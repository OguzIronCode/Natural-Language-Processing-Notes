from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Modeli ve Tokenizer'i yukle

model_name="gpt2"
tokenizer=GPT2Tokenizer.from_pretrained(model_name)
model=GPT2LMHeadModel.from_pretrained(model_name)

# Ornek Metin

text="I go to park to" 

# Tokenizasyon

inputs=tokenizer.encode(text, return_tensors="pt")

# Metin üretimi

outputs=model.generate(inputs, max_length=10)

# Sonucu decode etmemiz gerekiyor

generated_text=tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)