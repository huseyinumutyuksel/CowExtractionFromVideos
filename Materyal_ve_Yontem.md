# Materyal ve Yöntem

## 2. Materyal ve Yöntem

### 2.1. Veri Setinin Oluşturulması

Bu çalışmada kullanılan video veri seti, ticari bir sığır ahırsında gerçek yetiştiricilik koşulları altında kayıt altına alınmış ham görüntü materyalinden elde edilmiştir. Ham veri, farklı kamera sistemleri ve kayıt ekipmanlarıyla alınmış olduğundan birden fazla video formatı (.mp4, .avi, .mov vb.) ve birbirinden farklı dosya isimlendirme düzenleri içermekteydi. Veri setinin makine öğrenmesi iş akışlarına uygun bir yapıya kavuşturulabilmesi için sistematik bir ön işleme süreci tasarlanmış ve uygulanmıştır. Bu süreç; manuel içerik değerlendirmesi, format dönüşümü, standart isimlendirme ve otomatik video bölütleme olmak üzere dört ana aşamadan oluşmaktadır.

---

### 2.2. Manuel Değerlendirme ve Video Seçimi

Ham veri setindeki tüm videolar araştırmacı tarafından izlenerek içerik kalitesi açısından değerlendirilmiştir. Bu değerlendirmede; görüntü bulanıklığı, aşırı titreşim, yetersiz aydınlatma, çerçevede hayvan bulunmaması ve yinelenen sahne içeriği gibi kriterler dikkate alınmıştır. Değerlendirme sonucunda analiz için yeterli nitelikte olmayan kayıtlar veri setinden çıkarılmış, geriye kalan **105 video** çalışmanın temel girdisini oluşturmak üzere seçilmiştir.

---

### 2.3. Format Dönüşümü ve Standart İsimlendirme

Seçilen 105 video, iki aşamalı bir betik tabanlı ön işlemden geçirilmiştir.

#### 2.3.1. Standart İsimlendirme

Farklı kaynaklardan gelen videoların tutarsız ve bilgi taşımayan dosya adları, Python tabanlı bir betik aracılığıyla sistematik bir şemaya dönüştürülmüştür. Betik, hedef klasördeki tüm video dosyalarını uzantıdan bağımsız olarak algılamakta; dosyaları alfabetik sıraya göre düzenleyerek her birine üç basamaklı sıralı bir numara atamaktadır. Elde edilen isimlendirme şeması aşağıdaki biçimde tanımlanmıştır:

```
VID_{NNN}.{uzantı}   →   Örnek: VID_001.mp4, VID_002.avi, …, VID_105.mp4
```

Bu işlem, veri setinin tüm yaşam döngüsü boyunca video kayıtlarına tekrarlanabilir ve belirsizlikten uzak bir şekilde atıfta bulunulmasını sağlamaktadır.

#### 2.3.2. Video Format Dönüşümü

İsimlendirme işleminin ardından, farklı kapsayıcı formatlarındaki (özellikle `.mov`) videolar `ffmpeg` kütüphanesi kullanılarak `.mp4` formatına dönüştürülmüştür. Dönüştürme işlemi, işlem süresini en aza indirmek amacıyla iki kademeli bir strateji izlemiştir:

1. **Hızlı Yeniden Kapsayıcılama (Remux):** Kaynak videonun codec yapısı uyumluysa video ve ses akışları yeniden kodlanmadan yalnızca kapsayıcı formatı değiştirilmiştir (`vcodec=copy, acodec=copy`). Bu yöntem kayıpsız ve son derece hızlıdır.

2. **Tam Yeniden Kodlama (Transcode):** Remux işleminin başarısız olduğu durumlarda video, H.264 codec'i (`libx264`) ile CRF=18 kalite seviyesinde ve `medium` hız ön ayarında yeniden kodlanmış; ses ise AAC formatına dönüştürülmüştür. CRF değeri 0 (kayıpsız) ile 51 (en düşük kalite) arasında değişmekte olup 18, görsel açıdan kayıpsıza yakın kaliteyi temsil etmektedir.

Zaten `.mp4` formatında bulunan videolar ise meta verilerini koruyacak şekilde doğrudan hedef konuma kopyalanmıştır (`shutil.copy2`). Betik aynı zamanda daha önce işlenmiş dosyaları atlayan bir süreç yeniden başlatma (*resume*) mekanizması sunmaktadır. Dönüşüm süreci sonunda 105 videonun tamamı `.mp4` formatında ve `VID_001.mp4`–`VID_105.mp4` isimlendirme şemasıyla standartlaştırılmış olarak elde edilmiştir.

**Tablo 1.** Video format dönüştürme sürecinde kullanılan teknik parametreler.

| Parametre | Değer |
|---|---|
| Hedef video formatı | MP4 (.mp4) |
| Video codec (yeniden kodlama) | H.264 (libx264) |
| Sabit hız faktörü (CRF) | 18 |
| Kodlama hız ön ayarı | medium |
| Ses codec | AAC |
| Mevcut .mp4 işlemi | Doğrudan kopyalama (shutil.copy2) |
| .mov işlemi | Remux → başarısız ise Transcode |

---

### 2.4. Otomatik İnek Bölütleme ve Klip Üretimi

Format standartlaştırması tamamlanan videolar, her bir hayvana ait bağımsız video klipler üretmek amacıyla nesne tespiti ve takibi tabanlı otomatik bir bölütleme ardışık düzenine (pipeline) tabi tutulmuştur. Bu bölütleme işlemi Python programlama dili ve aşağıda açıklanan bileşenler kullanılarak gerçekleştirilmiştir.

#### 2.4.1. Nesne Tespiti ve Takibi

İnek tespiti için, COCO veri seti üzerinde önceden eğitilmiş **YOLOv8-Medium Segmentasyon** modeli (`yolov8m-seg.pt`) kullanılmıştır. Model, yalnızca COCO sınıf tanımlayıcısı 19'a karşılık gelen "inek" sınıfını hedef alacak şekilde yapılandırılmıştır. Güven eşiği 0,75 olarak belirlenmiş; bu değerin altında kalan tespitler işlem dışında bırakılmıştır. Seçilen model, standart nesne tespitinin ötesinde piksel düzeyinde örnek bölütlemesi (instance segmentation) yapabilmekte; bu özellik arka plan kaldırma aşamasında kritik bir rol oynamaktadır.

Kare bazında tespit sonuçlarının birden fazla video karesine tutarlı biçimde eşleştirilmesi amacıyla `persist=True` parametresiyle etkinleştirilen kalıcı takip (persistent tracking) kullanılmıştır. Bu mekanizma, her hayvanın tüm video boyunca özgün ve değişmeyen bir takip tanımlayıcısıyla (track ID) ilişkilendirilmesini sağlamaktadır.

**Tablo 2.** Nesne tespiti ve takip modülünde kullanılan parametreler.

| Parametre | Değer | Açıklama |
|---|---|---|
| Model | YOLOv8-Medium Segmentasyon | Piksel düzeyinde bölütleme destekli |
| Model boyutu | ~53 MB | Disk üzerindeki model ağırlık dosyası |
| Hedef sınıf (COCO ID) | 19 | "Cow" (İnek) sınıfı |
| Güven eşiği | 0,75 | Bu değerin altındaki tespitler reddedilir |
| Kalıcı takip | Etkin (persist=True) | Kareler arası tutarlı track ID ataması |

#### 2.4.2. Sınır Kutusu Düzeltme: Üstel Hareketli Ortalama

Gerçek zamanlı nesne tespitinde bağlayıcı kutu (bounding box) koordinatları kare kare dalgalanma eğilimi göstermektedir; bu durum çıktı videolarında göze çarpan titreşime yol açmaktadır. Bu sorunu gidermek amacıyla her takip kimliğine özel bir **Üstel Hareketli Ortalama (Exponential Moving Average, EMA)** filtresi uygulanmıştır. Filtre, en güncel gözleme daha az, önceki durağan pozisyona ise daha fazla ağırlık vererek bağlayıcı kutunun konumunu pürüzsüzleştirir. Matematiksel formülasyon aşağıda verilmiştir:

$$\hat{b}_t = \alpha \cdot b_t + (1 - \alpha) \cdot \hat{b}_{t-1}$$

Burada $b_t$ mevcut karedeki ham tespit kutusunu, $\hat{b}_{t-1}$ önceki karenin düzeltilmiş kutusunu ve $\alpha$ düzeltme katsayısını ifade etmektedir. $\alpha = 0{,}2$ olarak seçilmiş olup bu değer güçlü bir pürüzsüzleştirme sağlarken hayvanın gerçek hareket dinamiklerini yeterince takip etmektedir. Filtre, her yeni video işlenmeye başlandığında sıfırlanmaktadır.

#### 2.4.3. Kısmi Hayvan Filtresi

Görüntünün kenar bölgelerine denk gelen, dolayısıyla çerçeve dışına taşan ve anatomik bütünlüğü bozulmuş hayvanların veri setine dahil edilmesini önlemek amacıyla bir kenar filtresi uygulanmıştır. Ham tespit kutusunun herhangi bir koordinatının çerçeve sınırından 5 piksel veya daha az uzakta olması durumunda ilgili tespit o kare için devre dışı bırakılmaktadır. Bu kural, kısmi görünürlüklü ineklerin çıktı kliplerine karışmasının önüne geçmektedir.

#### 2.4.4. Kırpma ve Dolgu

Tespiti onaylanan hayvanın bağlayıcı kutusu, EMA pürüzsüzleştirme işleminin ardından her yönde **30 piksel** genişletilmektedir. Bu dolgu, tırnak, kuyruk ve kulak gibi uzuv uçlarının çerçeve dışında kalmasını engellemektedir. Genişletilmiş koordinatlar, taşma olmaması için görüntü sınırlarıyla kırpılmaktadır.

#### 2.4.5. Arka Plan Kaldırma

YOLOv8-Medium Segmentasyon modeli, sınır kutusuyla birlikte her hayvan için piksel düzeyinde bir segmentasyon maskesi üretmektedir. Bu maske kullanılarak çerçevedeki inek dışındaki tüm bölgeler **siyah (RGB: 0, 0, 0)** renkli bir arka planla değiştirilmiştir. Arka plan değiştirmede iki yöntem desteklenmekle birlikte bu çalışmada **yumuşak maskeleme (soft masking)** yöntemi tercih edilmiştir:

1. **İkili Maskeleme (Binary Masking):** Segmentasyon maskesi doğrudan uygulanır; inek ve arka plan arasında keskin bir geçiş elde edilir.
2. **Yumuşak Maskeleme (Soft Masking):** Segmentasyon maskesi önce 3×3 çekirdek boyutunda 2 iterasyon dilatasyon işlemine tabi tutulur; ardından 15×15 boyutunda Gaussian bulanıklaştırma uygulanır. Elde edilen yumuşak alfa kanalı aracılığıyla ön plan (inek) ve arka plan doğrusal olarak harmanlanır:

$$I_{çıktı} = I_{inek} \cdot \alpha + I_{arka\ plan} \cdot (1 - \alpha)$$

Bu yöntem, hayvan silüetinin kenarlarında doğal bir geçiş bölgesi oluşturarak pikselleşme ve sert kesim etkilerini önemli ölçüde azaltmaktadır.

**Tablo 3.** Arka plan kaldırma modülünde kullanılan parametreler.

| Parametre | Değer |
|---|---|
| Maskeleme yöntemi | Yumuşak maskeleme (soft) |
| Arka plan rengi | Siyah (R=0, G=0, B=0) |
| Dönüşüm çekirdeği | 3×3 piksel |
| Dönüşüm (dilation) iterasyonu | 2 |
| Gaussian bulanıklaştırma çekirdeği | 15×15 piksel |

#### 2.4.6. Çözünürlük Standardizasyonu

Her hayvana ait kırpılmış görüntü, sabit bir **640×640 piksel** çözünürlüğe getirilmektedir. Oransal bozulmayı önlemek amacıyla mektup kutusu (*letterboxing*) yöntemi uygulanmıştır: kırpılan bölge hedef boyuttan büyükse yalnızca küçültme yapılmakta, küçükse ölçekleme uygulanmamakta; her iki durumda da boşluk kalan alan siyah piksellerle doldurulmaktadır. Bu yaklaşım, farklı vücut ölçülerine sahip hayvanların görüntülerinin tutarsız ölçekleme nedeniyle bozulmasını engellemektedir.

#### 2.4.7. Video Yazımı ve Süre Filtresi

Her takip kimliğine ait kareler, kaynaktan alınan ve tam sayıya yuvarlanmış FPS değeri kullanılarak geçici bir `.mp4` dosyasına yazılmaktadır. FPS yuvarlama işlemi, bazı kamera sistemlerinden kaynaklanan kesirli FPS değerlerinin (örneğin 29,97 fps → 30 fps) video kapsayıcısı düzeyinde oluşturabileceği zaman tabanı hatalarını önlemektedir. Video işlemi tamamlandığında, her geçici dosyanın toplam süresi hesaplanmakta ve **4,0 saniyenin altındaki klipler** veri setinden çıkarılmaktadır. Bu eşik, yürüyüş döngüsünün gözlemlenemeyeceği kadar kısa kayıtların analize dahil edilmesini engellemektedir. Süre koşulunu sağlayan klipler `{kaynak}_{inek_no}.mp4` adlandırma şemasıyla kalıcı çıktı klasörüne taşınmaktadır.

**Tablo 4.** Çıktı video üretimi ve filtreleme parametreleri.

| Parametre | Değer |
|---|---|
| Çıktı çözünürlüğü | 640×640 piksel |
| Boyutlandırma yöntemi | Letterboxing (oransal küçültme + siyah dolgu) |
| Çıktı video formatı | MP4 (H.264) |
| FPS hesabı | Tam sayıya yuvarlama (round) |
| Minimum klip süresi | 4,0 saniye |
| Ek kenar boşluğu (dolgu) | 30 piksel (her yön) |
| Kenar filtresi (sınır payı) | 5 piksel |

---

### 2.5. Veri Seti İstatistikleri

Uygulanan ardışık düzen sonucunda elde edilen veri setine ilişkin sayısal bilgiler Tablo 5'te özetlenmiştir.

**Tablo 5.** Elde edilen veri setine ait genel istatistikler.

| Özellik | Değer |
|---|---|
| Giriş video sayısı | 105 |
| Toplam giriş süresi | ~127,5 dakika (7.652 saniye) |
| Toplam giriş karesi | 232.950 |
| Ortalama giriş video süresi | ~72,9 saniye |
| Giriş video çözünürlükleri | 1920×1080, 1280×720, 848×480 |
| Giriş video kare hızları | 29,14 – 30,00 fps (bir kayıt: 240,37 fps) |
| Çıktı klip sayısı | 765 |
| Toplam çıktı süresi | ~120,1 dakika (7.205 saniye) |
| Toplam çıktı karesi | 219.462 |
| Ortalama klip süresi | ~9,4 saniye |
| Minimum klip süresi | 4,0 saniye |
| Maksimum klip süresi | 49,7 saniye |
| Çıktı video çözünürlüğü | 640×640 piksel (tüm klipler) |
| Çıktı arka plan rengi | Siyah (RGB: 0, 0, 0) |
| Giriş başına ortalama inek klip sayısı | ~7,7 |
| En fazla klip üretilen video | 44 klip |
| Hiç klip üretilemeyen video sayısı | 5 |

**Tablo 6.** Çıktı kliplerinin süre dağılımı.

| Süre Aralığı | Klip Sayısı | Oran (%) |
|---|---|---|
| 4 – 10 saniye | 546 | 71,4 |
| 11 – 30 saniye | 203 | 26,5 |
| 31 saniye ve üzeri | 16 | 2,1 |
| **Toplam** | **765** | **100,0** |

---

### 2.6. Süreç Akış Şeması

Veri hazırlama sürecinin tüm aşamaları Şekil 1'de özetlenmektedir.

```
┌──────────────────────────────────────────────────────────────────────┐
│                        HAM VİDEO VERİSİ                             │
│         (karışık isimler, .mp4 / .avi / .mov / vb. formatlar)       │
└──────────────────────┬───────────────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────┐
         │  AŞAMA 1: Manuel Seçim  │
         │  105 video seçildi      │
         └────────────┬────────────┘
                      │
                      ▼
      ┌───────────────────────────────┐
      │  AŞAMA 2: İsimlendirme        │
      │  VID_001 … VID_105            │
      │  (alfabetik sıra, 3 basamak)  │
      └──────────────┬────────────────┘
                     │
                     ▼
      ┌───────────────────────────────────────┐
      │  AŞAMA 3: Format Dönüşümü             │
      │  .mov/.avi → .mp4                     │
      │  Remux (hızlı) veya                   │
      │  H.264 CRF=18 (yeniden kodlama)       │
      └──────────────┬────────────────────────┘
                     │
                     ▼
      ┌────────────────────────────────────────────────────────────┐
      │  AŞAMA 4: Otomatik Bölütleme (Her Video İçin)             │
      │                                                            │
      │  Kare oku → YOLOv8m-seg (conf≥0.75, sınıf=19)            │
      │       │                                                    │
      │       ├─ Kenar filtresi (sınır payı: 5px)                 │
      │       ├─ EMA pürüzsüzleştirme (α=0.20)                   │
      │       ├─ Dolgu ekleme (30px)                              │
      │       ├─ Yumuşak segmentasyon maskesi (siyah arka plan)   │
      │       ├─ Kırpma + Letterbox → 640×640 px                  │
      │       └─ Track ID başına geçici .mp4'e yaz                │
      │                                                            │
      │  Video sonu → Süre ≥ 4,0 sn ise kaydet, değilse sil      │
      └──────────────────────────────┬─────────────────────────────┘
                                     │
                                     ▼
              ┌──────────────────────────────────────┐
              │  ÇIKTI: 765 bireysel inek video klibi │
              │  640×640 px · siyah arka plan · ≥4 sn │
              └──────────────────────────────────────┘
```

**Şekil 1.** Veri hazırlama ardışık düzeninin akış şeması.

---

### 2.7. Görsel Örnekler

Şekil 2 ve Şekil 3, ardışık düzenin girdi ve çıktılarını karşılaştırmalı olarak sunmaktadır.

> **Şekil 2.** Ham giriş videolarından örnek kareler. Her karede birden fazla inek yer almakta ve görüntüler farklı çözünürlük ile kamera açısı koşullarını yansıtmaktadır.
> *(report_frames/input_VID_003.jpg, input_VID_005.jpg, input_VID_006.jpg, input_VID_013.jpg)*

> **Şekil 3.** Aynı kaynak videodan elde edilen dört farklı inek klibinin örnek kareleri. Her klip 640×640 piksel çözünürlüğünde ve siyah arka planlıdır; hayvanın dışındaki tüm piksel bilgisi kaldırılmıştır.
> *(report_frames/output_grid_VID013.jpg)*

> **Şekil 4.** Sekiz farklı kaynak videodan rastgele seçilmiş bireysel inek kliplerinden örnek kareler. Veri setinin hayvan ölçeği, duruş ve görüş açısı çeşitliliğini yansıtmaktadır.
> *(report_frames/output_mosaic_8cows.jpg)*

---

### 2.8. Yazılım Ortamı

Tüm işlemler Python programlama dili kullanılarak gerçekleştirilmiştir. Kullanılan başlıca kütüphaneler ve sürüm gereksinimleri Tablo 7'de sunulmaktadır.

**Tablo 7.** Kullanılan yazılım bileşenleri ve kullanılan sürümler.

| Bileşen | Kullanılan Sürüm | Kullanım Amacı |
|---|---|---|
| Python | 3.8+ | Genel programlama dili |
| ultralytics | 8.3.236 | YOLOv8 tespit, segmentasyon ve takip |
| opencv-python | 4.12.0.88 | Video okuma/yazma, görüntü işleme, maske uygulaması |
| numpy | 2.2.6 | Dizi ve maske hesaplamaları |
| torch | 2.9.1 | YOLOv8 arka uç derin öğrenme çerçevesi |
| torchvision | 0.24.1 | Görüntü dönüşüm yardımcı programları |
| tqdm | 4.67.1 | İlerleme gösterimi |
| ffmpeg-python | – | Format dönüşüm betiğinde FFmpeg Python sarmalayıcı |
| FFmpeg | Sistem bağımlı* | Video format dönüşümü (.mov → .mp4) |

*Format dönüşümü işlemi sırasında kullanılan FFmpeg binary sürümü, işlemin gerçekleştirildiği sistemde `ffmpeg -version` komutu çalıştırılarak doğrulanabilir. OpenCV 4.12.0.88 paketinin dahili FFmpeg bağımlılığı önceden derlenmiş ikili dosyalar (*prebuilt binaries*) olarak temin edilmektedir.
