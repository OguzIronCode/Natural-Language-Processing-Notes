#Metindeki fazla boşlukları temizleme

text1="Hello    world!     Oguzhan"

cleaned_text1=" ".join(text1.split())

print(cleaned_text1)

#Metindeki içindeki büyük harfleri küçük harfe çevirme

text2="Hello World OGUZHAN Welcome To 2035"

cleaned_text2=text2.lower()

print(cleaned_text2)


#Metindeki Noktalama İşaretlerini Kaldır

import string

text3="Hello Word! Oguzhan, Welcome To : 2035"

cleaned_text3=text3.translate(str.maketrans("","",string.punctuation))

print(cleaned_text3)

#Metindeki Özel Karakterleri Kaldırma
#Pythonda ki re(Düzenli İfadeler)(Regular Expression) kullanıyoruz

import re

text4="Hello Word #Oguzhan *Demirbas +2035"

cleaned_text4= re.sub(r"[^A-Za-z0-9\s]","",text4)

print(cleaned_text4)

#Metindeki Yazım Hatalarının Düzeltilmesi

from textblob import TextBlob

text5="Helıo Wirld Oguzhan Welcone To 2035"

cleaned_text5=str(TextBlob(text5).correct())

print(cleaned_text5)

#Metindeki HTML Ya da URL Etiketlerinin Düzeltilmesi

from bs4 import BeautifulSoup

text6="<div>Hello World Oguzhan Welcome To 2035</div>"

cleaned_text6=BeautifulSoup(text6,"html.parser").get_text()

print(cleaned_text6)

