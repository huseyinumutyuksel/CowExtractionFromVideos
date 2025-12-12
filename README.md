# Cow Extraction Project

Bu proje, bir videodaki inekleri tespit edip takip ederek her bir inek için ayrı bir video dosyası oluşturur. 
105 adet video içeren bir klasörden, her bir ineğin tek başına olduğu videoları (crop) `output_cows` klasörüne kaydeder.

## Özellikler

- **YOLOv8** kullanarak nesne tespiti ve takibi.
- **SOLID** prensiplerine uygun mimari.
- Özelleştirilebilir konfigürasyon (`config/settings.py`).
- Otomatik dosya isimlendirme (`cow_0001.mp4`, `cow_0002.mp4`...).

## Kurulum

1. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. Girdi videolarını hazırlayın:
   - Proje ana dizininde `input_videos` adında bir klasör oluşturun (veya `config/settings.py` dosyasından yolu değiştirin).
   - Videolarınızı bu klasöre koyun.

## Çalıştırma

Projeyi başlatmak için ana dizinde terminali açın ve şu komutu çalıştırın:

```bash
python main.py
```

Kod otomatik olarak çalışan bir klasördeki `input` videolarını tarayacak ve `output_cows` klasörüne sonuçları yazacaktır.

## Konfigürasyon

`config/settings.py` dosyasını düzenleyerek şunları değiştirebilirsiniz:
- Video kaynak klasörü (`INPUT_VIDEOS_DIR`)
- Çıktı klasörü (`OUTPUT_VIDEOS_DIR`)
- Kullanılan YOLO modeli (`YOLO_MODEL_NAME`)
- Güven eşiği (`CONFIDENCE_THRESHOLD`)

## Mimari

- `src/interfaces.py`: Soyut sınıflar (Interface Segregation, Dependency Inversion).
- `src/detector.py`: YOLO modelini sarmalar (Detector implementation).
- `src/writer.py`: Video yazma işlemlerini yönetir.
- `src/processor.py`: Ana iş mantığını içerir (Video okuma, crop, resize).
