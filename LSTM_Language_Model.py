# import libraries

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Creat Dataset

texts = [
    "Bugün hava çok güzel", "Dışarıda yağmur yağıyor", "Sabah erkenden kalktım", "Kahvaltıda omlet yedim",
    "İşe gitmek için hazırlandım", "Otobüs durağında bekliyorum", "Kahve içmek beni uyandırıyor", "Bugün çok işim var",
    "Toplantı beklediğimden uzun sürdü", "Öğle yemeğinde salata yedim", "Arkadaşımla kahve içtik", "Yeni bir kitap aldım",
    "Kitap okumak beni rahatlatıyor", "Akşam yemeği için yemek yaptım", "Televizyonda güzel bir film var", "Yarın erken kalkmam lazım",
    "Telefonumun şarjı azaldı", "Müzik dinlemeyi çok seviyorum", "Yürüyüşe çıkmak iyi geldi", "Bugün kendimi yorgun hissediyorum",
    "Yeni bir kıyafet aldım", "Evde temizlik yapmam gerekiyor", "Marketten eksikleri tamamladım", "Hava bugün biraz soğuk",
    "En sevdiğim yemek makarna", "Bilgisayarım biraz yavaşladı", "İnternet bağlantısı kesildi", "Yeni bir diziye başladım",
    "Spor yapmak enerji veriyor", "Bugün çok güldük", "Eski bir dostu gördüm", "Gelecek hafta tatile çıkıyorum",
    "Deniz kenarında oturmak huzurlu", "Bugün biraz başım ağrıyor", "İşlerimi vaktinde bitirdim", "Akşam annemi aradım",
    "Kedi videoları çok komik", "Sabah kahvaltısını hiç atlamam", "Bugün kendimi şanslı hissediyorum", "Yarın hava güneşli olacakmış",
    "Ders çalışmak bazen sıkıcı", "Yeni bir dil öğrenmek istiyorum", "Pencereden dışarıyı izliyorum", "Sessiz bir ortam arıyorum",
    "Bugün trafikte çok bekledim", "Alışveriş yapmak beni mutlu ediyor", "Yeni bir tarif denedim", "Yemek çok lezzetli olmuş",
    "Bugün biraz geç uyandım", "Hafta sonu planım yok", "Evde dinlenmek istiyorum", "Bugün çok verimli geçti",
    "Yarın toplantım var", "Yeni bir kalem aldım", "Notlarımı düzenlemem lazım", "Bugün biraz dalgınım",
    "Güneş gözlüğümü evde unuttum", "Dışarıda rüzgar esiyor", "Akşam yürüyüşü çok keyifli", "Bugün kendime zaman ayırdım",
    "Çay içmek beni sakinleştiriyor", "Yeni bir hedef belirledim", "Başarmak için çalışmak lazım", "Bugün hava kapalı",
    "Kış mevsimini çok severim", "Yarın için heyecanlıyım", "Yeni bir ayakkabı aldım", "Ayakkabılarım çok rahat",
    "Bugün işe geç kaldım", "Alarmı duymamışım", "Hızlıca giyindim", "Kahvaltıyı yolda yaptım",
    "Bugün gökyüzü çok mavi", "Kuşların sesini duyuyorum", "Bahçedeki çiçekler çok güzel", "Doğada vakit geçirmeliyim",
    "Bugün biraz stresliyim", "Derin nefes alıyorum", "Her şey yoluna girecek", "Kendime güveniyorum",
    "Bugün yeni bir şey öğrendim", "Öğrenmek hiç bitmiyor", "Bilgi paylaştıkça çoğalır", "Kitaplarım benim hazinem",
    "Bugün akşam erken yatacağım", "Uykumu almam gerekiyor", "Rüyalarımı hatırlıyorum", "Sabah mutlu uyandım",
    "Bugün çok mesaj aldım", "Telefonum hiç susmadı", "Sosyal medyada çok vakit geçirdim", "Teknoloji hayatımızı kolaylaştırıyor",
    "Bugün biraz hüzünlüyüm", "Eski günleri özledim", "Zaman çok çabuk geçiyor", "Anın tadını çıkarmalıyım",
    "Bugün yemekte balık var", "Limonlu salata harika", "Tatlı yemeyi çok seviyorum", "Meyve tabağı hazırladım",
    "Bugün hava mis gibi", "Parkta çocuklar oynuyor", "Bisiklet sürmek çok eğlenceli", "Dışarıda hayat var",
    "Bugün ödevlerimi bitirdim", "Kütüphaneye gittim", "Yeni kaynaklar buldum", "Araştırma yapmak önemli",
    "Bugün evde film izledik", "Mısır patlattık", "Gülmekten yorulduk", "Arkadaşlık çok değerli",
    "Bugün yeni bir uygulama denedim", "İşimi çok kolaylaştırdı", "Dijital dünya çok hızlı", "Kendimi güncelliyorum",
    "Bugün biraz sessizim", "Kendi içime döndüm", "Düşünmek iyi geliyor", "Kararlarımı gözden geçirdim",
    "Bugün hava rüzgarlı", "Uçurtma uçurmak isterdim", "Çocukluğumu hatırladım", "O günler çok güzeldi",
    "Bugün iş yerinde kutlama vardı", "Pasta yedik", "Herkes çok mutluyum", "Ekip ruhu önemli",
    "Bugün spor salonuna gittim", "Ağırlık çalıştım", "Vücudum güçleniyor", "Sağlıklı yaşam şart",
    "Bugün markete gittim", "Fiyatlar biraz artmış", "İndirimleri takip ediyorum", "Ekonomi önemli konu",
    "Bugün balkon yıkadım", "Çiçekleri suladım", "Evim çok temiz oldu", "Huzur burada",
    "Bugün şarkı söyledim", "Sesim çok güzel değil ama olsun", "Müzik ruhun gıdasıdır", "Dans etmeyi seviyorum",
    "Bugün birine yardım ettim", "Teşekkür aldım", "İyilik yapmak harika", "Dünya daha güzel olacak",
    "Bugün hava çok sıcak", "Dondurma yemek iyi geldi", "Soğuk su içiyorum", "Yaz geldi sonunda",
    "Bugün yeni bir defter aldım", "Günlük tutmaya başladım", "Yazmak beni rahatlatıyor", "Düşüncelerimi kağıda döküyorum",
    "Bugün çok kalabalıktı", "Şehir hayatı yorucu", "Biraz sakinlik istiyorum", "Köye gitmek hayalim",
    "Bugün güneş battı", "Ay dede çıktı", "Geceyi çok seviyorum", "Sessizlik hakim",
    "Bugün kod yazdım", "Hataları düzelttim", "Program çalışıyor", "Başardım",
    "Bugün kendimi iyi hissediyorum", "Gülümsemek bedava", "Hayat güzel", "Şükretmek lazım",
    "Bugün akşam dışarı çıkacağız", "Güzel giyinmeliyim", "Parfüm sürdüm", "Hazırım",
    "Bugün bir karar verdim", "Artık daha düzenli olacağım", "Planlı yaşamak güzel", "Zamanımı iyi kullanmalıyım",
    "Bugün hava biraz yağmurlu", "Şemsiyemi yanıma aldım", "Yağmurda yürümek romantik", "Toprak kokusu harika",
    "Bugün mutfağı topladım", "Bulaşıkları yıkadım", "Her yer parlıyor", "Düzenli ev huzurlu ev",
    "Bugün resim yaptım", "Renkler çok canlı", "Sanatla uğraşmak güzel", "Yaratıcı olmalıyım",
    "Bugün çok yoruldum", "Ayaklarım ağrıyor", "Biraz dinlenmeliyim", "Koltukta uzanıyorum",
    "Bugün yeni bir saat aldım", "Zamanı takip ediyorum", "Randevularıma sadığım", "Disiplin önemli",
    "Bugün gömleğimi ütüledim", "Jilet gibi oldu", "Temiz giyinmek lazım", "Özsaygı önemlidir",
    "Bugün çok düşündüm", "Hayatın anlamı nedir", "Küçük mutluluklar yeter", "Büyük hayaller kuruyorum",
    "Bugün kendime kahve yaptım", "Köpüğü çok bol oldu", "Kitabımı açtım", "Keyif yapıyorum",
    "Bugün hava serin", "Ceketimi giydim", "Sonbahar geliyor", "Yapraklar dökülüyor",
    "Bugün çiçekçiye gittim", "Bir demet gül aldım", "Odam çok güzel kokuyor", "Canlılık geldi",
    "Bugün çok sabırlıydım", "Kimseye kızmadım", "Hoşgörü her şeydir", "Barış içinde yaşamalıyız",
    "Bugün radyo dinledim", "Eski şarkılar çaldı", "Nostalji yaptım", "Anılar canlandı",
    "Bugün işten erken çıktım", "Kendime ödül verdim", "Sinemaya gittim", "Film çok sürükleyiciydi",
    "Bugün biraz kararsızdım", "Hangi ayakkabıyı giysem", "Sonunda siyah olanı seçtim", "Şık oldum",
    "Bugün telefonum sessizde kalmış", "Aramaları görmemişim", "Geri dönüş yaptım", "İletişim koptu bir an",
    "Bugün kahvaltıda bal yedim", "Çok tatlıydı", "Enerji topladım", "Güne hazırım",
    "Bugün saçlarımı kestirdim", "Yeni tarzım çok güzel", "Değişiklik iyi gelir", "Kendimi yeniledim",
    "Bugün çok kitap okudum", "Yüz sayfa bitti", "Hikaye çok heyecanlı", "Elimden bırakamıyorum",
    "Bugün hava biraz sisli", "Yollar görünmüyor", "Dikkatli sürmek lazım", "Emniyet kemeri takılı",
    "Bugün balkonda oturdum", "Komşularla selamlaştım", "Mahalle kültürü güzel", "İnsanları seviyorum",
    "Bugün bir rüya gördüm", "Uçuyordum gökyüzünde", "Çok gerçekçiydi", "Keşke bitmeseydi",
    "Bugün çantamı hazırladım", "Yarın yolculuk var", "Valizimi kontrol ettim", "Her şey tamam",
    "Bugün çok soru sordum", "Cevapları merak ediyorum", "Merak ilmin hocasıdır", "Araştırmaya devam",
    "Bugün biraz acele ettim", "Otobüsü ucu ucuna yakaladım", "Nefes nefese kaldım", "Zamanlama önemli",
    "Bugün hava çok berrak", "Dağlar bile görünüyor", "Manzara harika", "Fotoğraf çektim",
    "Bugün kek pişirdim", "Evi mis gibi koku sardı", "Komşuya da verdim", "Paylaşmak güzeldir",
    "Bugün çok dua ettim", "Kalbim huzur doldu", "İnanmak güç verir", "Yalnız değilim",
    "Bugün biraz sinirliydim", "Sonra sakinleştim", "Öfke kontrolü şart", "Kırmamak lazım kimseyi",
    "Bugün yeni bir defter bitirdim", "Anılarla doldu içi", "Yazmak şifadır", "Geleceğe miras",
    "Bugün hava dondurucu", "Atkımı sıkıca sardım", "Eldivenlerimi giydim", "Sıcak eve girmeliyim",
    "Bugün meyve suyu sıktım", "Portakal ve havuç karışık", "C vitamini deposu", "Sağlığımı koruyorum",
    "Bugün çok fazla düşündüm", "Gelecek neler getirecek", "Umutlu olmalıyım", "Yarın yeni bir başlangıç",
    "Bugün kendimi ödüllendirdim", "En sevdiğim tatlıyı yedim", "Mutluluk bu kadar basit", "Kendimi seviyorum",
    "Bugün bilgisayarı güncelledim", "Yeni özellikler geldi", "Hızlandı biraz", "Teknoloji güzel şey",
    "Bugün yolda kedi gördüm", "Süt verdim ona", "Miyavlayarak teşekkür etti", "Hayvanları koruyalım",
    "Bugün çok verimli çalıştım", "Bütün maddeleri işaretledim", "Liste bitti", "Huzurla uyuyabilirim",
    "Bugün hava tam gezmelik", "Dışarı attım kendimi", "Adım sayımı tamamladım", "Hareket berekettir",
    "Bugün çok sustum", "Dinlemek daha öğretici", "Anlamak için susmak lazım", "İnsanları gözlemledim",
    "Bugün bir hata yaptım", "Özür diledim hemen", "Hatasız kul olmaz", "Önemli olan ders almak",
    "Bugün hava çok temiz", "Ciğerlerim bayram etti", "Orman havası gibi", "Şehir dışına çıktım",
    "Bugün biraz geç kaldım", "Önemli değil dediler", "Anlayışlı insanlar iyi ki var", "Mutlu oldum",
    "Bugün yemekte çorba var", "Sıcak sıcak içtim", "İçim ısındı", "Kışın en güzeli bu",
    "Bugün bir şarkıya takıldım", "Dilimden düşmüyor", "Sürekli mırıldanıyorum", "Ritmik bir gün",
    "Bugün ayna karşısında konuştum", "Kendime moral verdim", "Sen yaparsın dedim", "Motivasyon şart",
    "Bugün hava çok bulutlu", "Belki kar yağar", "Bekliyoruz heyecanla", "Beyaz örtü özlendi",
    "Bugün bir çocukla oynadım", "Dünya onun gözünde masal", "Masumiyet çok güzel", "Çocuk kalabilmeli insan",
    "Bugün çok fazla çay içtim", "Uykum kaçabilir", "Kitap okumaya devam o zaman", "Gece uzun",
    "Bugün biraz yorgunum ama mutluyum", "İşler bitti çünkü", "Başarı hissi harika", "Dinlenmeyi hak ettim",
    "Bugün hava açtı sonunda", "Güneş yüzünü gösterdi", "İçim aydınlandı", "Gülümseme sebebi",
    "Bugün yeni bir arkadaş edindim", "Çok ortak noktamız var", "Sohbet çok akıcıydı", "Yeni insanlar yeni dünyalar",
    "Bugün kendime bir söz verdim", "Asla pes etmeyeceğim", "Zorluklar beni güçlendirir", "Yoluma devam ediyorum",
    "Bugün hava biraz nemli", "Saçlarım kabardı", "Önemli değil gülümsüyorum", "Hayat her haliyle güzel",
    "Bugün çok güldüm yine", "Gözümden yaş geldi", "Kahkaha en iyi ilaç", "Dostlarla olmak şans",
    "Bugün sessizce oturdum", "Sadece nefesimi dinledim", "Farkındalık kazandım", "Buradayım ve hayattayım",
    "Bugün bir hediye aldım", "Beklemiyordum hiç", "Sürprizler hayatın tuzu biberi", "Çok özel hissettim",
    "Bugün hava çok sakin", "Yaprak bile kımıldamıyor", "Dinginlik ruhuma işledi", "Dingin bir gün",
    "Bugün çok iş hallettim", "Banka işleri bitti", "Faturaları ödedim", "Sorumluluklar yerine getirildi",
    "Bugün bir hayal kurdum", "Dünyayı geziyordum", "Belki bir gün gerçek olur", "Hayal kurmak bedava",
    "Bugün hava çok kararsız", "Bir güneş bir yağmur", "Aynı hayat gibi", "Her şeye hazırlıklıyım",
    "Bugün bir yazı yazdım", "Duygularımı anlattım", "Rahatladım sanki", "Kelime büyülüdür",
    "Bugün çok bekledim ama değdi", "Sonuç mükemmel", "Sabır acıdır ama meyvesi tatlıdır", "Mutluyum şimdi",
    "Bugün hava çok parlak", "Gözlerim kamaştı", "Hayat ışık saçıyor", "Pozitif bakıyorum",
    "Bugün bir iyilik gördüm", "İnsanlığa inancım arttı", "Hala güzel insanlar var", "Umut hep var",
    "Bugün hava çok serindi", "Battaniyeye sarıldım", "Sıcak çikolata içtim", "Kış keyfi başladı",
    "Bugün bir resme baktım uzun uzun", "Çok şey anlattı bana", "Sanat evrenseldir", "Ruhum beslendi",
    "Bugün çok koştum", "Otobüse yetişmek için", "Spor oldu bana da", "Hızlı bir sabah",
    "Bugün hava çok tatlı", "Tam yürüyüş havası", "Sahile indim", "Deniz kokusunu içime çektim",
    "Bugün bir kuş kondu pencereme", "Ekmek verdim ona", "Neşeyle öttü", "Küçük bir misafir",
    "Bugün çok sabırlı davrandım", "Zor bir durumu yönettim", "Kendimle gurur duyuyorum", "Olgunlaşıyorum galiba",
    "Bugün hava çok puslu", "Göz gözü görmüyor", "Gizemli bir hava", "Evde oturmak en iyisi",
    "Bugün bir çiçek ektim", "Can suyunu verdim", "Büyümesini bekleyeceğim", "Emek vermek güzel",
    "Bugün çok mutlu uyandım", "Sebepsiz bir neşe", "Belki de güzel bir gün olacak", "Pozitif enerji",
    "Bugün hava çok ferah", "Yağmurdan sonra her yer temiz", "Taze bir başlangıç", "Yenilenmiş hissediyorum",
    "Bugün bir fıkra duydum", "Çok komikti", "Herkesle paylaştım", "Gülmek paylaştıkça güzel",
    "Bugün çok sakin bir gün", "Hiç gürültü yok", "Kafa dinledim iyice", "Ruhum dinlendi",
    "Bugün hava çok dramatik", "Kara bulutlar şimşekler", "Doğanın gücü inanılmaz", "Saygı duyuyorum",
    "Bugün bir söz okudum", "Beni çok etkiledi", "Hayat felsefem olabilir", "Derin düşünceler",
    "Bugün çok çabuk geçti", "Zaman nasıl aktı anlamadım", "Dolu dolu yaşadım", "Mutlu bir akşam",
    "Bugün hava çok nazlı", "Güneş bir görünüyor bir kaçıyor", "Oyun oynuyor bizimle", "Eğlenceli bir gün",
    "Bugün bir karar aldım", "Sağlıklı besleneceğim artık", "Vücuduma iyi bakmalıyım", "Yeni hayat başlıyor",
    "Bugün çok şanslı bir gündü", "Her işim rast gitti", "Mucizelere inanırım", "Teşekkürler hayat",
    "Bugün hava çok asil", "Gri ve vakur", "Hüzünlü ama dik", "Aynı ben gibi",
    "Bugün bir şarkı besteledim", "Melodisi çok duygusal", "İçimden geldiği gibi", "Müzik benim dilim",
    "Bugün çok fazla uyumuşum", "Rüya üstüne rüya", "Dinlenmiş uyanmak harika", "Pazar keyfi",
    "Bugün hava çok sıcak ama rüzgarlı", "En sevdiğim hava tipi", "Hem sıcak hem serin", "Dengeli bir gün",
    "Bugün bir mektup yazdım", "Eski usul kağıt kalem", "Zarfa koyup gönderdim", "Heyecanla cevap bekliyorum",
    "Bugün çok fazla meyve yedim", "Vitamin deposu oldum", "Vücudum canlandı", "Doğal beslenme en iyisi",
    "Bugün hava çok ciddi", "Sanki bir şey söyleyecek", "Doğayı dinliyorum", "Sessiz mesajlar",
    "Bugün bir oyun oynadım", "Çocuklarla birlikte", "Ben de çocuk oldum sanki", "Neşe dolu saatler",
    "Bugün çok fazla konuşmadım", "Sadece gülümsedim", "Bazen sessizlik en iyi cevaptır", "Huzurlu bir tavır",
    "Bugün hava çok berrak ve soğuk", "Tam bir kış günü", "Güneş var ama ısıtmıyor", "Taze bir hava",
    "Bugün bir hata yaptım ama düzelttim", "Öğrenme süreci devam ediyor", "Kendime kızmıyorum", "Gelişiyorum",
    "Bugün çok güzel bir gün", "Her şey yolunda", "Kendimi çok iyi hissediyorum", "Mutluluk yakın",
    "Bugün hava çok gizemli", "Sis her yeri kaplamış", "Masal dünyası gibi", "Büyüleyici bir atmosfer",
    "Bugün bir iyilik yaptım gizlice", "Kimse bilmesin istedim", "İçim huzurla doldu", "Gerçek iyilik bu",
    "Bugün çok fazla çalıştım ama değdi", "Proje bitti sonunda", "Büyük bir yük kalktı", "Rahat bir nefes",
    "Bugün hava çok dostane", "Ilık ve yumuşak", "İnsanı kucaklıyor sanki", "Sevgi dolu bir gün",
    "Bugün bir fidan diktim", "Geleceğe bir nefes", "Dünya için küçük bir adım", "Mutlu bir eylem",
    "Bugün çok güldüm yine arkadaşlarımla", "Anılar biriktirdik", "Hayat onlarla güzel", "Canım arkadaşlarım",
    "Bugün hava çok hüzünlü bugün", "Yağmur damlaları camda", "Eski şarkılar eşliğinde", "Duygusal bir gün",
    "Bugün bir kitap bitirdim harikaydı", "Ufkum açıldı resmen", "Yeni dünyalar keşfettim", "Okumak yaşamaktır",
    "Bugün çok sakin bir akşam", "Kitabım ve kahvem", "Bundan daha iyisi olamaz", "Huzurun adresi"
]


# Tokenizer and Preparation Of Series

tokenizer= Tokenizer()
tokenizer.fit_on_texts(texts)
total_words=len(tokenizer.word_index)+1
print(total_words)

# Text Sorting and Padding 

input_sequences= []
for text in texts :
    token_list=tokenizer.texts_to_sequences([text])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)
        
max_sequence_length = max (len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding= "pre")

X,y =input_sequences[:,:-1], input_sequences[:,-1]

y= tf.keras.utils.to_categorical(y,num_classes=total_words)


# Creat LSTM Model

model=Sequential()
model.add(Embedding(total_words,50,input_length=X.shape[1]))
model.add(LSTM(100,return_sequences=False))
model.add(Dense(total_words,activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train LSTM  Model

model.fit(X,y,epochs= 200,verbose=1)

# Evaluation and Text Completion

def predict_next_word(seed_text,next_words):
    
    for _ in range (next_words):
        
       token_list= tokenizer.texts_to_sequences([seed_text])[0]
       token_list=pad_sequences([token_list],maxlen= max_sequence_length-1, padding="pre")
       predicted_probs=model.predict(token_list,verbose=0)
       predicted_word_index=np.argmax(predicted_probs,axis=-1)
       predicted_word=tokenizer.index_word[predicted_word_index[0]]
       seed_text=seed_text+ " "+predicted_word
       
    return seed_text

seed_text= "Bugün hava"

print(predict_next_word(seed_text,5))