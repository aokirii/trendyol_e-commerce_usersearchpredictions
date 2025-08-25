import pandas as pd
import os


def analyze_parquet_files():
    """
    Parquet dosyalarını analiz eder ve eksik değerleri tablo halinde gösterir
    """

    # Dosya yolları
    file_paths = {
        'Fashion_search_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/fashion_search_log.parquet',
        'Fashion_sitewide_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/fashion_sitewide_log.parquet',
        'metadata': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/metadata.parquet',
        'search_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/search_log.parquet',
        'sitewide_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/sitewide_log.parquet',
        'top_terms_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/top_terms_log.parquet'
    }

    print("VERİ ANALİZ RAPORU")
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
                    'Eksik_Yüzde': round(missing_percent, 2)
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
    analyze_parquet_files()