import pandas as pd
import os


def analyze_product_data():
    """
    ENGLISH:
    Analyzes product data and displays only the missing values in a tabular format.

    TÜRKÇE:
    Ürün verilerini analiz eder ve sadece eksik değerleri tablo halinde gösterir
    """

    # Dosya yolları
    file_paths = {
        'metadata': 'YOUR FILE PATH',
        'price_rate_review_data': 'YOUR FILE PATH',
        'search_log': 'YOUR FILE PATH',
        'sitewide_log': 'YOUR FILE PATH',
        'top_terms_log': 'YOUR FILE PATH'
    }

    print("ÜRÜN VERİ ANALİZ RAPORU")
    print("=" * 80)

    all_data = []

    # Her dosyayı analiz et
    for file_name, file_path in file_paths.items():
        try:
            if not os.path.exists(file_path):
                print(f"❌ {file_name}: Dosya bulunamadı")
                continue

            df = pd.read_parquet(file_path)

            # Her sütun için bilgi topla
            for column in df.columns:
                missing_count = df[column].isnull().sum()
                missing_percent = (missing_count / len(df)) * 100

                all_data.append({
                    'Dosya': file_name,
                    'Sütun': column,
                    'Veri_Tipi': str(df[column].dtype),
                    'Toplam_Satır': len(df),
                    'Eksik_Sayı': missing_count,
                    'Eksik_Yüzde': round(missing_percent, 2),
                    'Benzersiz_Değer': df[column].nunique()
                })

        except Exception as e:
            print(f"❌ {file_name}: Hata - {str(e)}")

    # Sonuçları tablo halinde göster
    if all_data:
        result_df = pd.DataFrame(all_data)
        print("\nEKSİK DEĞER ANALİZ TABLOSU")
        print("=" * 80)
        print(result_df.to_string(index=False))
    else:
        print("Hiçbir dosya analiz edilemedi.")


if __name__ == "__main__":
    analyze_product_data()
