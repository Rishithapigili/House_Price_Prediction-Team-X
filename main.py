from model import HousePriceModel


def display_menu():
    """Display the main menu."""
    print("\n" + "=" * 60)
    print("       HOUSE PRICE PREDICTION SYSTEM")
    print("         (KC House Dataset — Linear Regression)")
    print("=" * 60)
    print("  1. Load & Display Dataset Info")
    print("  2. Select Feature Columns")
    print("  3. Train Model & Display R-squared Score")
    print("  4. Predict House Price for New Input")
    print("  5. View Prediction History")
    print("  6. Exit")
    print("=" * 60)


def main():
    """Main function — menu-driven interaction loop."""
    model = HousePriceModel()

    while True:
        display_menu()
        choice = input("  Enter your choice (1-6): ").strip()

        if choice == '1':
            model.display_dataset_info()

        elif choice == '2':
            model.select_features()

        elif choice == '3':
            model.train_model()

        elif choice == '4':
            model.predict_price()

        elif choice == '5':
            model.load_predictions()

        elif choice == '6':
            print("\n  Thank you for using House Price Prediction System!")
            print("  Goodbye!\n")
            break

        else:
            print("\n  [!] Invalid choice. Please enter a number between 1 and 6.")


if __name__ == '__main__':
    main()
