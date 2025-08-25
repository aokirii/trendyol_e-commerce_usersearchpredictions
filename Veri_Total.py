import pandas as pd
import os
from datetime import datetime


def save_analysis_to_csv(all_data, analysis_type="analiz", output_folder="output"):
    """
    Analiz sonuçlarını CSV dosyasına kaydeder

    Args:
        all_data: Analiz sonuçlarının bulunduğu liste
        analysis_type: Analiz türü (örn: "kullanici", "urun", "arama_terimi")
        output_folder: Çıktı klasörü
    """

    # Çıktı klasörünü oluştur
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 '{output_folder}' klasörü oluşturuldu")

    # DataFrame oluştur
    result_df = pd.DataFrame(all_data)

    # Dosya adı oluştur (tarih ile)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{analysis_type}_analiz_{timestamp}.csv"
    filepath = os.path.join(output_folder, filename)

    # CSV'ye kaydet
    result_df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"✅ Analiz sonuçları kaydedildi: {filepath}")
    print(f"📊 Toplam {len(result_df)} satır kaydedildi")

    return filepath


# KULLANICI VERİLERİ İÇİN GÜNCELLENMIŞ FONKSİYON
def analyze_and_save_user_data():
    """
    Kullanıcı verilerini analiz eder ve CSV'ye kaydeder
    """

    file_paths = {
        'Fashion_search_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/fashion_search_log.parquet',
        'Fashion_sitewide_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/fashion_sitewide_log.parquet',
        'metadata': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/metadata.parquet',
        'search_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/search_log.parquet',
        'sitewide_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/sitewide_log.parquet',
        'top_terms_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/user/top_terms_log.parquet'
    }

    print("KULLANICI VERİ ANALİZ RAPORU")
    print("=" * 80)

    all_data = []

    for file_name, file_path in file_paths.items():
        try:
            if not os.path.exists(file_path):
                print(f"❌ {file_name}: Dosya bulunamadı")
                continue

            df = pd.read_parquet(file_path)

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

        # CSV'ye kaydet
        save_analysis_to_csv(all_data, "kullanici_verileri")
    else:
        print("Hiçbir dosya analiz edilemedi.")


# ARAMA TERİMİ VERİLERİ İÇİN GÜNCELLENMIŞ FONKSİYON
def analyze_and_save_search_data():
    """
    Arama terimi verilerini analiz eder ve CSV'ye kaydeder
    """

    file_path = '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/term/search_log.parquet'

    print("ARAMA TERİMİ VERİ ANALİZ RAPORU")
    print("=" * 80)

    try:
        if not os.path.exists(file_path):
            print(f"❌ Dosya bulunamadı: {file_path}")
            return

        df = pd.read_parquet(file_path)

        analysis_data = []
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            missing_percent = (missing_count / len(df)) * 100

            analysis_data.append({
                'Dosya': 'search_log',
                'Sütun': column,
                'Veri_Tipi': str(df[column].dtype),
                'Toplam_Satır': len(df),
                'Eksik_Sayı': missing_count,
                'Eksik_Yüzde': round(missing_percent, 2),
                'Benzersiz_Değer': df[column].nunique()
            })

        # Sonuçları göster
        result_df = pd.DataFrame(analysis_data)
        print("\nEKSİK DEĞER ANALİZ TABLOSU")
        print("=" * 80)
        print(result_df.to_string(index=False))

        # CSV'ye kaydet
        save_analysis_to_csv(analysis_data, "arama_terimi_verileri")

    except Exception as e:
        print(f"❌ Hata: {str(e)}")


# ÜRÜN VERİLERİ İÇİN GÜNCELLENMIŞ FONKSİYON
def analyze_and_save_product_data():
    """
    Ürün verilerini analiz eder ve CSV'ye kaydeder
    """

    file_paths = {
        'metadata': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/content/metadata.parquet',
        'price_rate_review_data': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/content/price_rate_review_data.parquet',
        'search_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/content/search_log.parquet',
        'sitewide_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/content/sitewide_log.parquet',
        'top_terms_log': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/content/top_terms_log.parquet'
    }

    print("ÜRÜN VERİ ANALİZ RAPORU")
    print("=" * 80)

    all_data = []

    for file_name, file_path in file_paths.items():
        try:
            if not os.path.exists(file_path):
                print(f"❌ {file_name}: Dosya bulunamadı")
                continue

            df = pd.read_parquet(file_path)

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

    if all_data:
        result_df = pd.DataFrame(all_data)
        print("\nEKSİK DEĞER ANALİZ TABLOSU")
        print("=" * 80)
        print(result_df.to_string(index=False))

        # CSV'ye kaydet
        save_analysis_to_csv(all_data, "urun_verileri")
    else:
        print("Hiçbir dosya analiz edilemedi.")


# OTURUM VERİLERİ İÇİN GÜNCELLENMIŞ FONKSİYON
def analyze_and_save_session_data():
    """
    Oturum verilerini analiz eder ve CSV'ye kaydeder
    """

    file_paths = {
        'train_sessions': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/train_sessions.parquet',
        'test_sessions': '/Users/kasimesen/Desktop/Veltrix/Kodlar/data/test_sessions.parquet'
    }

    print("OTURUM VERİ ANALİZ RAPORU")
    print("=" * 80)

    all_data = []

    for file_name, file_path in file_paths.items():
        try:
            if not os.path.exists(file_path):
                print(f"❌ {file_name}: Dosya bulunamadı")
                continue

            df = pd.read_parquet(file_path)

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

    if all_data:
        result_df = pd.DataFrame(all_data)
        print("\nEKSİK DEĞER ANALİZ TABLOSU")
        print("=" * 80)
        print(result_df.to_string(index=False))

        # CSV'ye kaydet
        save_analysis_to_csv(all_data, "oturum_verileri")
    else:
        print("Hiçbir dosya analiz edilemedi.")


def run_all_analyses():


    print("🚀 TÜM VERİ ANALİZLERİ BAŞLATIYOR...")
    print("=" * 80)

    print("\n1️⃣ Kullanıcı Verileri Analizi:")
    print("-" * 50)
    analyze_and_save_user_data()

    print("\n2️⃣ Arama Terimi Analizi:")
    print("-" * 50)
    analyze_and_save_search_data()

    print("\n3️⃣ Ürün Verileri Analizi:")
    print("-" * 50)
    analyze_and_save_product_data()

    print("\n4️⃣ Oturum Verileri Analizi:")
    print("-" * 50)
    analyze_and_save_session_data()

    print("\n" + "=" * 80)
    print("✅ TÜM ANALİZLER TAMAMLANDI!")
    print("📁 Sonuçlar 'output' klasöründe CSV dosyaları olarak kaydedildi.")
    print("=" * 80)

if __name__ == "__main__":
    run_all_analyses()
