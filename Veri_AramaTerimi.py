import pandas as pd
import os


def analyze_search_log():
    """
    ENGLISH:
    Analyzes the search-term log file and displays only the missing values in a table.
    
    TÜRKÇE:
    Arama terimi log dosyasını analiz eder ve sadece eksik değerleri tablo halinde gösterir
    """

    file_path = 'YOUR FILE PATH'

    print("ARAMA TERİMİ VERİ ANALİZ RAPORU")
    print("=" * 80)

    try:
        if not os.path.exists(file_path):
            print(f"❌ Dosya bulunamadı: {file_path}")
            return

        df = pd.read_parquet(file_path)

        # Tablo için veri hazırla
        analysis_data = []
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100

            analysis_data.append({
                'Sütun': column,
                'Veri_Tipi': str(df[column].dtype),
                'Toplam_Satır': len(df),
                'Eksik_Sayı': missing_count,
                'Eksik_Yüzde': round(missing_percent, 2),
                'Benzersiz_Değer': df[column].nunique()
            })

        # Sonuçları tablo halinde göster
        result_df = pd.DataFrame(analysis_data)
        print("\nEKSİK DEĞER ANALİZ TABLOSU")
        print("=" * 80)
        print(result_df.to_string(index=False))

    except Exception as e:
        print(f"❌ Hata: {str(e)}")


if __name__ == "__main__":
    analyze_search_log()
