import pandas as pd
import os
from datetime import datetime


def save_analysis_to_csv(all_data, analysis_type="analiz", output_folder="output"):
    """
    ENGLISH:
    Saves analysis results to a CSV file.

    Args:
        all_data: List containing the analysis results.
        analysis_type: Type of analysis (e.g., "user", "product", "search_term").
        output_folder: Output folder.

    TÜRKÇE:
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
        'Fashion_search_log': 'YOUR FILE PATH',
        'Fashion_sitewide_log': 'YOUR FILE PATH',
        'metadata': 'YOUR FILE PATH',
        'search_log': 'YOUR FILE PATH',
        'sitewide_log': 'YOUR FILE PATH',
        'top_terms_log': 'YOUR FILE PATH'
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

    file_path = 'YOUR FILE PATH'

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
        'metadata': 'YOUR FILE PATH',
        'price_rate_review_data': 'YOUR FILE PATH',
        'search_log': 'YOUR FILE PATH',
        'sitewide_log': 'YOUR FILE PATH',
        'top_terms_log': 'YOUR FILE PATH'
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
        'train_sessions': 'YOUR FILE PATH',
        'test_sessions': 'YOUR FILE PATH'
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
