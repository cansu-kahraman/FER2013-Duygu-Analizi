# FER2013-Duygu-Analizi

accuracy: 0.1572 - loss: 3.2894 Test Doğruluğu: 15.14%

Bu projenin temel amacı, yapay sinir ağları kullanarak yüz ifadelerine dayalı duygu analizi
gerçekleştirmektir. Projede, model performansını etkileyen hiperparametrelerin (örneğin, gizli
katman sayısı, nöron sayısı, aktivasyon fonksiyonu, öğrenme oranı ve momentum oranı)
değişiminin modelin doğruluk ve kayıp değerlerine etkisi incelenecektir. Bu analizle, modelin
farklı hiperparametre ayarlarında nasıl davranış sergilediği raporlanacaktır ve en iyi
doğruluğu sağlayan hiperparametre kombinasyonu belirlenecektir.

Gizli Katman Sayıları
Model, bir giriş katmanı, 4 gizli katman ve bir çıkış katmanından oluşmaktadır.
Conv2D: 3 adet (32, 64, 128 filtreli).
Dense: 1 adet (256 nöronlu yoğun bağlantılı katman).
Çıkış Katmanı: Dense(7, softmax)

Gizli Katman Nöron Sayıları
Evrişimsel Katmanlar:
İlk katman: 32 filtre (Conv2D(32, ...))
İkinci katman: 64 filtre (Conv2D(64, ...))
Üçüncü katman: 128 filtre (Conv2D(128, ...))
Yoğun Bağlantılı Katmanlar:
256 nöron (Dense(256, activation='relu', ...)).

Aktivasyon Fonksiyonları
Gizli katmanlarda ReLU aktivasyon fonksiyonu, çıkış katmanında ise softmax fonksiyonu kullanılmıştır.

Öğrenme Oranı ve Momentum Oranı
sgd_optimizer = SGD(learning_rate=0.01, momentum=0.9)

