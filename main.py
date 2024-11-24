import sys
import os
import re
import pandas as pd
import sqlite3
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget, QTableWidget,
    QTableWidgetItem, QFileDialog, QLabel, QMessageBox, QTabWidget, QPushButton
)
from referens import compare_vcf
from claster import analyze_mutation_files


class RegressionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OLS Regression")
        self.setGeometry(100, 100, 800, 600)

        self.initUI()

        # Connect to SQLite database
        self.db_connection = sqlite3.connect("regression_data.db")
        self.cursor = self.db_connection.cursor()
        self.create_tables()

    def initUI(self):
        self.tabs = QTabWidget()

        # Tab 1: Main regression interface
        self.tab1 = self.create_regression_tab()
        self.tabs.addTab(self.tab1, "Regression")

        # Tab 2:
        self.tab2 = self.create_visualization_tab()
        self.tabs.addTab(self.tab2, "Analysis")

        self.setCentralWidget(self.tabs)

    def create_tables(self):
        """Create tables for each variable if they don't already exist."""
        variables = [
            'сумма температур больше 10С', 'сумма осадков, мм', 'рН KCl',
            'Орг. В-во %', 'P2O5', 'Nлегкогидр', 'К2О', 'Зависимая переменная'
        ]
        for var in variables:
            self.cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS `{var}` (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    value REAL
                )
            """)
        self.db_connection.commit()

    def save_to_db(self, df):
        """Save data to the database."""
        try:
            for column in df.columns:
                if column in ['сумма температур больше 10С', 'сумма осадков, мм', 'рН KCl',
                              'Орг. В-во %', 'P2O5', 'Nлегкогидр', 'К2О', 'Зависимая переменная']:
                    for value in df[column]:
                        self.cursor.execute(f"INSERT INTO `{column}` (value) VALUES (?)", (value,))
            self.db_connection.commit()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save data to database: {e}")

    def create_regression_tab(self):
        """Create the main regression tab."""
        layout = QVBoxLayout()

        # Buttons
        self.load_button = QPushButton("Load Data")
        self.load_button.clicked.connect(self.load_data)
        layout.addWidget(self.load_button)

        self.calculate_button = QPushButton("Calculate Regression")
        self.calculate_button.clicked.connect(self.calculate_regression)
        self.calculate_button.setEnabled(False)
        layout.addWidget(self.calculate_button)

        # Table
        self.table = QTableWidget()
        layout.addWidget(self.table)

        # Summary Label
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        container = QWidget()
        container.setLayout(layout)
        return container

    def create_visualization_tab(self):
        """Create a tab for data visualization and display of summary and TSV data."""
        layout = QVBoxLayout()

        # Button for visualizing data
        self.visualize_button = QPushButton("Visualize Data")
        self.visualize_button.clicked.connect(self.visualize_data)
        layout.addWidget(self.visualize_button)

        # Table for displaying summary and TSV data
        self.data_table = QTableWidget()
        layout.addWidget(self.data_table)

        container = QWidget()
        container.setLayout(layout)
        return container

    def visualize_data(self):
        # Выбор файлов и папки для сохранения результатов
        vcf_file, _ = QFileDialog.getOpenFileName(self, "Select VCF File", "", "VCF Files (*.vcf)")
        if not vcf_file:
            QMessageBox.warning(self, "Warning", "VCF file not selected.")
            return

        tsv_file, _ = QFileDialog.getOpenFileName(self, "Select TSV File", "", "TSV Files (*.tsv)")
        if not tsv_file:
            QMessageBox.warning(self, "Warning", "TSV file not selected.")
            return

        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            QMessageBox.warning(self, "Warning", "Output directory not selected.")
            return

        try:
            # Анализ VCF файла и сохранение summary
            compare_vcf(vcf_file, output_dir)

            homozygous_file = output_dir + "/homozygous_summary.txt"
            heterozygous_file = output_dir + "/heterozygous_summary.txt"
            output_file = f"{output_dir}/mutation_clusters.txt"
            analyze_mutation_files(homozygous_file, heterozygous_file, output_file, n_clusters=3)

            data = self.load_mutation_data(output_dir + "/mutation_clusters.txt")

            # Если данные загружены успешно, выполняем регрессию для разных типов мутаций
            if data is not None:
                years = [2017, 2022]  # Список лет
                # Регрессионный анализ для гетерозиготных мутаций (кластер 1)
                regression_results_hetero = self.regression_analysis_for_cluster_and_mutation(data, cluster_id=1,
                                                                                              years=years,
                                                                                              mutation_type='Heterozygous')
                # Регрессионный анализ для гомозиготных мутаций (кластер 2)
                regression_results_homo = self.regression_analysis_for_cluster_and_mutation(data, cluster_id=2,
                                                                                            years=years,
                                                                                            mutation_type='Homozygous')

        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def load_mutation_data(self, file_path):
        """
        Загружает данные из файла mutation_clusters.txt.
        """
        try:
            data = pd.read_csv(file_path, sep='\t')
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def regression_analysis_for_cluster_and_mutation(self, data, cluster_id, years, mutation_type):
        """
        Строит регрессионный анализ для данных по выбранному кластеру, году и типу мутации.
        """
        try:
            # Фильтрация данных по кластеру и типу мутации
            filtered_data = data[(data['Cluster'] == cluster_id) & (data['MutationType'] == mutation_type)]

            if filtered_data.empty:
                print(f"No data found for cluster {cluster_id} and mutation type {mutation_type}.")
                return None

            # Убедимся, что годы представлены как строки, и подготовим данные для регрессионного анализа
            years_str = [str(year) for year in years]  # Преобразуем года в строки
            X = np.array(years_str).reshape(-1, 1)  # Года как независимая переменная
            y = filtered_data[years_str].mean(axis=1)  # Среднее значение по годам для мутаций

            # Проверка на пустые значения
            if np.any(np.isnan(y)):
                print(f"Error: Some mutation values are NaN.")
                return None

            # Строим модель линейной регрессии
            model = LinearRegression()
            model.fit(X, y)

            # Вывод результатов
            coef = model.coef_
            intercept = model.intercept_
            score = model.score(X, y)  # Коэффициент детерминации (R^2)

            print(f"Cluster {cluster_id} - MutationType: {mutation_type}")
            print(f"  Coefficients: {coef}")
            print(f"  Intercept: {intercept}")
            print(f"  R^2: {score}")

            # Визуализация результатов
            plt.figure(figsize=(10, 6))
            plt.plot(years_str, y, 'bo-', label=f"Data: {mutation_type}")
            plt.plot(years_str, model.predict(X), 'r-', label=f"Regression Line")
            plt.xlabel('Year')
            plt.ylabel('Mean Mutations')
            plt.title(f'Regression for {mutation_type} mutations (Cluster {cluster_id})')
            plt.legend()
            plt.show()

            return {
                'coef': coef,
                'intercept': intercept,
                'score': score
            }
        except Exception as e:
            print(f"Error during regression analysis: {e}")
            return None

    def load_data(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Excel File", "", "Excel Files (*.xlsx)")
        if file_path:
            try:
                # Load data from Excel
                self.df = pd.read_excel(file_path, engine="openpyxl")
                self.populate_table(self.df)
                self.save_to_db(self.df)
                self.calculate_button.setEnabled(True)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load data: {e}")

    def populate_table(self, df):
        self.table.setRowCount(len(df))
        self.table.setColumnCount(len(df.columns))
        self.table.setHorizontalHeaderLabels(df.columns)

        for i, row in df.iterrows():
            for j, value in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(value)))

    def calculate_regression(self):
        try:
            X = self.df[['сумма температур больше 10С',
                         'сумма осадков, мм', 'рН KCl',
                         'Орг. В-во %',
                         'P2O5',
                         'Nлегкогидр',
                         'К2О']]
            X = sm.add_constant(X)
            y = self.df['Зависимая переменная']

            model = sm.OLS(y, X)
            results = model.fit()

            coeffs = results.params

            self.df['Расчетная зависимая переменная'] = (
                    coeffs['сумма температур больше 10С'] * self.df['сумма температур больше 10С'] +
                    coeffs['сумма осадков, мм'] * self.df['сумма осадков, мм'] +
                    coeffs['рН KCl'] * self.df['рН KCl'] +
                    coeffs['Орг. В-во %'] * self.df['Орг. В-во %'] +
                    coeffs['P2O5'] * self.df['P2O5'] +
                    coeffs['Nлегкогидр'] * self.df['Nлегкогидр'] +
                    coeffs['К2О'] * self.df['К2О'] +
                    coeffs['const']).round(2)

            self.populate_table(self.df)
            self.summary_label.setText(f"Regression Summary:\n{results.summary()}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to calculate regression: {e}")

    def closeEvent(self, event):
        """Close the database connection on app exit."""
        self.db_connection.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = RegressionApp()
    main_window.show()
    sys.exit(app.exec_())
