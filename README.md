# Cow Extraction Project / İnek Çıkarma Projesi

[English](#english) | [Türkçe](#türkçe)

---

<a name="english"></a>
## 🇬🇧 English

### Overview
This project detects and tracks cows in a video, creating a separate video file for each individual cow. It processes videos from a source folder and saves cropped videos of isolated cows into the `output_cows` folder.

### Features
- **YOLOv8** for object detection and tracking.
- **SOLID** principles-compliant architecture.
- Customizable configuration (`config/settings.py`).
- Automatic file naming (`cow_0001.mp4`, `cow_0002.mp4`...).

### Installation

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

2. Prepare input videos:
   - Create a folder named `input_videos` in the project root (or change the path in `config/settings.py`).
   - Place your videos in this folder.

### Usage

To start the project, open a terminal in the root directory and run:

```bash
python main.py
```

The code will automatically scan `input` videos and save the results to the `output_cows` folder.

### Configuration

You can edit `config/settings.py` to change:
- Video source directory (`INPUT_VIDEOS_DIR`)
- Output directory (`OUTPUT_VIDEOS_DIR`)
- YOLO model used (`YOLO_MODEL_NAME`)
- Confidence threshold (`CONFIDENCE_THRESHOLD`)
- Background color (`BACKGROUND_COLOR` - set to black by default to minimize distortions).

### Architecture
- `src/interfaces.py`: Abstract classes (Interface Segregation, Dependency Inversion).
- `src/detector.py`: Wraps YOLO model (Detector implementation).
- `src/writer.py`: Handles video writing operations.
- `src/processor.py`: Contains main business logic (Video reading, crop, resize).

---

<a name="türkçe"></a>
## 🇹🇷 Türkçe

### Genel Bakış
Bu proje, bir videodaki inekleri tespit edip takip ederek her bir inek için ayrı bir video dosyası oluşturur. 105 adet video içeren bir klasörden, her bir ineğin tek başına olduğu videoları (crop) `output_cows` klasörüne kaydeder.

### Özellikler

- **YOLOv8** kullanarak nesne tespiti ve takibi.
- **SOLID** prensiplerine uygun mimari.
- Özelleştirilebilir konfigürasyon (`config/settings.py`).
- Otomatik dosya isimlendirme (`cow_0001.mp4`, `cow_0002.mp4`...).

### Kurulum

1. Gereksinimleri yükleyin:
   ```bash
   pip install -r requirements.txt
   ```

2. Girdi videolarını hazırlayın:
   - Proje ana dizininde `input_videos` adında bir klasör oluşturun (veya `config/settings.py` dosyasından yolu değiştirin).
   - Videolarınızı bu klasöre koyun.

### Çalıştırma

Projeyi başlatmak için ana dizinde terminali açın ve şu komutu çalıştırın:

```bash
python main.py
```

Kod otomatik olarak çalışan bir klasördeki `input` videolarını tarayacak ve `output_cows` klasörüne sonuçları yazacaktır.

### Konfigürasyon

`config/settings.py` dosyasını düzenleyerek şunları değiştirebilirsiniz:
- Video kaynak klasörü (`INPUT_VIDEOS_DIR`)
- Çıktı klasörü (`OUTPUT_VIDEOS_DIR`)
- Kullanılan YOLO modeli (`YOLO_MODEL_NAME`)
- Güven eşiği (`CONFIDENCE_THRESHOLD`)
- Arka plan rengi (`BACKGROUND_COLOR` - bozulmaları gizlemek için varsayılan olarak siyahtır).

### Mimari

- `src/interfaces.py`: Soyut sınıflar (Interface Segregation, Dependency Inversion).
- `src/detector.py`: YOLO modelini sarmalar (Detector implementation).
- `src/writer.py`: Video yazma işlemlerini yönetir.
- `src/processor.py`: Ana iş mantığını içerir (Video okuma, crop, resize).
