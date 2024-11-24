from collections import defaultdict
import os

def parse_vcf(vcf_file):
    """Парсит VCF файл и возвращает список записей в формате (хромосома, позиция, референс, альтернативный аллель, образцы)."""
    variants = []
    with open(vcf_file, 'r') as file:
        header = []
        for line in file:
            if line.startswith('#'):
                if line.startswith('#CHROM'):
                    header = line.strip().split('\t')[9:]  # Сохраняем имена образцов
                continue  # Пропускаем комментарии
            parts = line.strip().split('\t')
            chrom = parts[0]
            pos = int(parts[1])  # Позиция (начинается с 1)
            ref = parts[3]  # Референсный аллель
            alt = parts[4].split(',') if parts[4] != '.' else []  # Альтернативные аллели
            samples = parts[9:]  # Остальные части - это образцы
            variants.append((chrom, pos, ref, alt, samples))
    return header, variants


def save_summary(output_dir, homozygous_data, heterozygous_data):
    """Сохраняет summary файлы для гомозиготных и гетерозиготных мутаций."""
    os.makedirs(output_dir, exist_ok=True)

    homozygous_file = os.path.join(output_dir, "homozygous_summary.txt")
    heterozygous_file = os.path.join(output_dir, "heterozygous_summary.txt")

    def save_to_file(file_path, data, title):
        with open(file_path, 'w') as file:
            file.write(f"{'Chromosome':<15}{'Mutations':<15}{'Sample'}\n")
            file.write("=" * 50 + "\n")

            # Сортировка по именам образцов
            sorted_data = sorted(data.items(), key=lambda x: x[0][1])  # Сортируем по имени образца (второй элемент)

            for ((chrom, sample_id), total_mutations) in sorted_data:
                file.write(f"{chrom:<15}{total_mutations:<15}{sample_id}\n")
        print(f"{title} сохранен: {file_path}")

    save_to_file(homozygous_file, homozygous_data, "Гомозиготный файл")
    save_to_file(heterozygous_file, heterozygous_data, "Гетерозиготный файл")


def compare_vcf(vcf_file, output_dir):
    """Анализирует VCF файл и сохраняет данные отдельно для гомозиготных и гетерозиготных мутаций."""
    header, variants = parse_vcf(vcf_file)
    homozygous_data = defaultdict(int)  # Для подсчета гомозиготных мутаций
    heterozygous_data = defaultdict(int)  # Для подсчета гетерозиготных мутаций

    for chrom, pos, ref, alt, samples in variants:
        for sample_id, sample_data in zip(header, samples):
            if sample_data == '.':
                continue  # Пропускаем отсутствующие данные

            # Извлекаем аллели
            alleles = sample_data.split(':')[0]  # Берем только аллели
            alleles = alleles.split('/') if '/' in alleles else alleles.split('|')

            # Пропускаем только случай 0/0
            if len(alleles) == 2 and alleles[0] == '0' and alleles[1] == '0':
                continue

            if len(alleles) == 2:
                if alleles[0] == alleles[1]:  # Если аллели одинаковы
                    homozygous_data[(chrom, sample_id)] += 1
                else:  # Если аллели разные
                    heterozygous_data[(chrom, sample_id)] += 1

    print("Гомозиготные данные:", homozygous_data)  # Отладка
    print("Гетерозиготные данные:", heterozygous_data)  # Отладка

    # Сохраняем summary файлы
    save_summary(output_dir, homozygous_data, heterozygous_data)
    print(f"Результаты сохранены в папке {output_dir}.")
