import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


# Функция для чтения данных из файла summary
def read_summary_file(file_path, mutation_type):
    # Чтение данных с разделителем: пробел или табуляция
    df = pd.read_csv(file_path, sep=r'\s+', header=0, engine='python')  # \s+ для обработки пробелов

    # Очистка имен столбцов
    df.columns = df.columns.str.strip()

    # Убираем строки с NaN в критичных столбцах
    df = df.dropna(subset=['Chromosome', 'Mutations', 'Sample'])

    # Убедимся, что столбец 'Mutations' содержит только числовые данные
    df['Mutations'] = pd.to_numeric(df['Mutations'], errors='coerce')
    df = df.dropna(subset=['Mutations'])  # Убираем строки с некорректными значениями в 'Mutations'

    # Добавляем столбец с типом мутации
    df['MutationType'] = mutation_type

    return df


# Обработка данных по гомозиготным и гетерозиготным файлам
def prepare_combined_data(homozygous_file, heterozygous_file):
    # Читаем гомозиготные данные
    homozygous_df = read_summary_file(homozygous_file, mutation_type='Homozygous')

    # Читаем гетерозиготные данные
    heterozygous_df = read_summary_file(heterozygous_file, mutation_type='Heterozygous')

    # Объединяем данные
    combined_df = pd.concat([homozygous_df, heterozygous_df], ignore_index=True)

    return combined_df


# Группировка данных по хромосомам и типу мутаций
def prepare_data_by_chromosome_and_type(df):
    # Группируем данные по хромосомам и типу мутации, подсчитываем общее количество мутаций
    df_grouped = df.groupby(['Chromosome', 'MutationType'])['Mutations'].sum().reset_index()

    # Нормализация данных
    scaler = StandardScaler()
    df_grouped['Mutations_scaled'] = scaler.fit_transform(df_grouped[['Mutations']])

    return df_grouped


# Кластеризация хромосом
def perform_kmeans_clustering(df_grouped, n_clusters=3):
    # Кластеризация по количеству мутаций (для каждого типа отдельно)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_grouped['Cluster'] = kmeans.fit_predict(df_grouped[['Mutations_scaled']])

    return df_grouped, kmeans


# Визуализация кластеров
def visualize_clusters(df_grouped):
    # Отдельная визуализация для каждого типа мутации
    for mutation_type in df_grouped['MutationType'].unique():
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='Chromosome',
            y='Mutations',
            hue='Cluster',
            data=df_grouped[df_grouped['MutationType'] == mutation_type],
            palette="viridis"
        )
        plt.title(f"{mutation_type} Clusters by Mutation Count")
        plt.xlabel("Chromosome")
        plt.ylabel("Total Mutations")
        plt.xticks(rotation=90)
        plt.legend(title="Cluster")
        plt.show()


# Сохранение результатов кластеризации
def save_cluster_results(df_grouped, output_file):
    # Сохраняем таблицу в файл
    df_grouped.to_csv(output_file, index=False, sep='\t')
    print(f"Результаты сохранены в файл: {output_file}")


# Основной метод
def analyze_mutation_files(homozygous_file, heterozygous_file, output_file, n_clusters=3):
    # Объединяем данные
    combined_df = prepare_combined_data(homozygous_file, heterozygous_file)

    # Группируем данные по хромосомам и типу мутации
    df_grouped = prepare_data_by_chromosome_and_type(combined_df)

    # Выполняем кластеризацию
    df_clustered, _ = perform_kmeans_clustering(df_grouped, n_clusters=n_clusters)

    # Визуализируем результаты
    visualize_clusters(df_clustered)

    # Сохраняем результаты
    save_cluster_results(df_clustered, output_file)


# Запуск анализа
if __name__ == "__main__":
    homozygous_file = "files/homozygous_summary.txt"
    heterozygous_file = "files/heterozygous_summary.txt"
    output_file = "mutation_clusters.txt"

    analyze_mutation_files(homozygous_file, heterozygous_file, output_file, n_clusters=3)
