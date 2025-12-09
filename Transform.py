from transformers import AutoTokenizer,AutoModel
import torch

# Model ve Tokenizer Yükleme

model_name="bert-base-uncased"
tokenizer=AutoTokenizer.from_pretrained(model_name)
model=AutoModel.from_pretrained(model_name)


# Metni tanımla

text="Transformes are amazing for natural Language processing."


# Metni tokenlara dönüştürme

inputs= tokenizer(text,return_tensors="pt")

 
# Modeli kullanarak metin temsilleri oluştur

with torch.no_grad(): #Back propagation iptal edildi(Modelimizin ağırlıkları güncellenmeyecek)
    outputs=model(**inputs)
    
    
# Çıkışlardan ilk tokenları alalım  

last_hidden_state=outputs.last_hidden_state
first_token_embedding= last_hidden_state[0,0,:].numpy()

print("Metin Temsili : İlk Token :")
print(first_token_embedding)