# main.py
from train import train_model
from visualize import plot_two_models


def train_menu(model_type):
    print("\n1. Fine-tuning Adam\n"
          "2. Fine-tuning AMSGrad\n"
          "3. Training from scratch Adam\n"
          "4. Main menu\n")
    choice = input()

    match choice:
        case "1":
            train_model(pretrained=True, model_path=model_type[0], lr=5e-5, epochs=30, use_amsgrad=False)
        case "2":
            train_model(pretrained=True, model_path=model_type[1], lr=5e-5, epochs=30, use_amsgrad=True)
        case "3":
            train_model(pretrained=False, model_path=model_type[2], lr=5e-5, epochs=30, use_amsgrad=False)
        case "4":
            return
        case _:
            print("Некорректный выбор")


if __name__ == "__main__":

    model_path = [
        "models/fine_tuning_adam.pth",
        "models/fine_tuning_amsgrad.pth",
        "models/scratch_adam.pth"
    ]

    DATA_ROOT = "data/Dataset/Images"


    while True:
        print("\n1. Train models\n"
              "2. Compare Adam vs AMSGrad\n"
              "3. Compare Scratch vs Fine-tuning\n"
              "4. Exit\n")

        x = input()

        match x:
            case "1":
                train_menu(model_path)
            case "2":
                plot_two_models(
                    "CSV/fine_tuning_adam_training_history.csv",
                    "CSV/fine_tuning_amsgrad_training_history.csv"
                )
            case "3":
                plot_two_models(
                    "CSV/fine_tuning_adam_training_history.csv",
                    "CSV/scratch_adam_training_history.csv"
                )
            case "4":
                break
            case _:
                print("Некорректный выбор")

