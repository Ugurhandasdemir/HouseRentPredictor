# 🏠 House Rent Predictor

Bu proje, Hindistan'daki büyük şehirlerdeki ev kiralama verilerini kullanarak kira fiyatlarını tahmin etmeye yönelik bir makine öğrenmesi uygulamasıdır. Projede veri ön işleme, görselleştirme, aykırı değer temizliği, özellik mühendisliği ve model karşılaştırmaları yer almaktadır.

## 🚀 Kullanılan Teknolojiler

- **Python**
- **Pandas**, **NumPy** – Veri işleme ve analiz
- **Scikit-learn** – Modelleme ve değerlendirme
- **XGBoost** – Gelişmiş regresyon modeli
- **Matplotlib**, **Seaborn** – Görselleştirme

## 🔍 Modeller

- **XGBoost Regressor** (GridSearchCV ile optimize edilmiştir)
- **Linear Regression**

Her iki model de log dönüşümlü kira (`log1p(rent)`) üzerinden eğitilmiş ve değerlendirilmiştir.

## 🗂️ Veri Seti

Proje, Hindistan'daki çeşitli şehirlerde (Mumbai, Delhi, Bangalore, Hyderabad, Chennai, Kolkata) ev kiralama verilerini içeren bir veri seti kullanmaktadır.

Veri kümesinde yer alan bazı sütunlar:

- `BHK`: Oda sayısı  
- `Size`: Evin metrekaresi  
- `Floor`: Kat bilgisi (bulunduğu kat ve toplam kat)  
- `Bath`: Banyo sayısı  
- `Furnishing Status`: Evin eşyalı durumu  
- `City`, `Area Type`, `Point of Contact`: Diğer özellikler  
- `Rent`: Tahmin edilmesi gereken hedef değer  


## 📊 Görseller

### 1. Şehre Göre Kira Dağılımı
![Şehre Göre Kira](https://github.com/Ugurhandasdemir/HouseRentPredictor/blob/main/plots/01_city_rent_distribution.png)

### 2. Korelasyon Matrisi
![Korelasyon Matrisi](https://github.com/Ugurhandasdemir/HouseRentPredictor/blob/main/plots/02_correlation_heatmap.png)

### 3. XGBoost - Gerçek vs Tahmin
![XGBoost Tahmin](https://github.com/Ugurhandasdemir/HouseRentPredictor/blob/main/plots/03_xgboost_predicted_vs_actual.png)

### 4. Linear Regression - Artıklar
![Linear Hatalar](github.com/Ugurhandasdemir/HouseRentPredictor/blob/main/plots/04_linear_regression_residuals.png)

### 5. Linear Regression - Gerçek vs Tahmin
![Linear Tahmin](https://github.com/Ugurhandasdemir/HouseRentPredictor/blob/main/plots/05_linear_regression_predicted_vs_actual.png)

### 6. XGBoost Regressor Sonuçları
![XGBoost](https://github.com/Ugurhandasdemir/HouseRentPredictor/blob/main/plots/XGBoost%20Regressor%20Sonu%C3%A7lar%C4%B1.png)

### 7. Linear Regression
![Linear](https://github.com/Ugurhandasdemir/HouseRentPredictor/blob/main/plots/Linear%20Regression.png)
