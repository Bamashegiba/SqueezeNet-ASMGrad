import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_two_models(csv_path_1, csv_path_2):
    """
    Строит один комбинированный график с 4 метриками для двух моделей.
    CSV должны содержать столбцы:
    lr_val, loss_val, accuracy_val, precision_val, recall_val
    """

    # Загружаем данные
    df1 = pd.read_csv(csv_path_1)
    df2 = pd.read_csv(csv_path_2)

    # Создаём папку для графиков
    os.makedirs("PNG", exist_ok=True)

    epochs1 = range(1, len(df1) + 1)
    epochs2 = range(1, len(df2) + 1)

    model1_name = csv_path_1[4:-21]
    model2_name = csv_path_2[4:-21]
    # model1_name = os.path.basename(csv_path_1).replace(".csv", "")
    # model2_name = os.path.basename(csv_path_2).replace(".csv", "")

    # -----------------------------
    # ОБЪЕДИНЁННЫЙ ГРАФИК
    # -----------------------------
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Сравнение моделей: {model1_name} vs {model2_name}", fontsize=16)

    metrics = ["accuracy_val", "loss_val", "precision_val", "recall_val"]
    titles = ["Accuracy", "Loss", "Precision", "Recall"]

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = divmod(i, 2)
        ax[row][col].plot(epochs1, df1[metric], label=model1_name, linewidth=2)
        ax[row][col].plot(epochs2, df2[metric], label=model2_name, linewidth=2)
        ax[row][col].set_title(title)
        ax[row][col].set_xlabel("Epoch")
        ax[row][col].grid(True, linestyle="--", alpha=0.5)
        ax[row][col].legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # чтобы заголовок не перекрывался
    filename = f"PNG/{model1_name}_VS_{model2_name}_combined.png"
    plt.savefig(filename)
    plt.show()
    plt.close()

    print(f"Комбинированный график сохранён: {filename}")

