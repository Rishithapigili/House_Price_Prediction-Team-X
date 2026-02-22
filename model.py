import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from datetime import datetime


class HousePriceModel:
    """House Price Prediction model using Linear Regression on KC House dataset."""

    DEFAULT_FEATURES = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'yr_built'
    ]

    def __init__(self, csv_path='kc_house_data.csv', data_file='data.txt'):
        self.csv_path = csv_path
        self.data_file = data_file
        self.df = None
        self.model = None
        self.features = list(self.DEFAULT_FEATURES)
        self.r2 = None
        self.is_trained = False

    # ------------------------------------------------------------------ #
    #  Dataset operations
    # ------------------------------------------------------------------ #
    def load_dataset(self):
        """Load the KC House dataset from CSV."""
        self.df = pd.read_csv(self.csv_path)
        return self.df

    def display_dataset_info(self):
        """Print basic information about the loaded dataset."""
        if self.df is None:
            self.load_dataset()
        print("\n" + "=" * 60)
        print("            KC HOUSE DATASET INFORMATION")
        print("=" * 60)
        print(f"  Rows    : {self.df.shape[0]}")
        print(f"  Columns : {self.df.shape[1]}")
        print("-" * 60)
        print("  Column Names:")
        for i, col in enumerate(self.df.columns, 1):
            print(f"    {i:>2}. {col}")
        print("-" * 60)
        print("  First 5 rows:")
        print(self.df.head().to_string(index=False))
        print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Feature selection
    # ------------------------------------------------------------------ #
    def get_numeric_columns(self):
        """Return numeric columns excluding 'id', 'date', and 'price'."""
        if self.df is None:
            self.load_dataset()
        exclude = {'id', 'date', 'price'}
        return [c for c in self.df.select_dtypes(include='number').columns if c not in exclude]

    def select_features(self):
        """Interactive feature selection — user picks from available numeric columns."""
        available = self.get_numeric_columns()
        print("\n" + "=" * 60)
        print("           FEATURE SELECTION")
        print("=" * 60)
        print("  Available numeric columns:")
        for i, col in enumerate(available, 1):
            marker = " *" if col in self.DEFAULT_FEATURES else ""
            print(f"    {i:>2}. {col}{marker}")
        print("  (* = default feature)")
        print("-" * 60)
        print("  Enter column numbers separated by commas")
        print("  (or press Enter to use defaults):")
        user_input = input("  >> ").strip()

        if user_input == "":
            self.features = list(self.DEFAULT_FEATURES)
            print(f"\n  Using default features ({len(self.features)} selected).")
        else:
            try:
                indices = [int(x.strip()) for x in user_input.split(',')]
                selected = [available[i - 1] for i in indices if 1 <= i <= len(available)]
                if not selected:
                    print("  No valid columns selected — falling back to defaults.")
                    self.features = list(self.DEFAULT_FEATURES)
                else:
                    self.features = selected
                    print(f"\n  Selected features: {self.features}")
            except (ValueError, IndexError):
                print("  Invalid input — using default features.")
                self.features = list(self.DEFAULT_FEATURES)

        self.is_trained = False
        print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Training
    # ------------------------------------------------------------------ #
    def train_model(self, test_size=0.2, random_state=42):
        """Train Linear Regression and compute R-squared score."""
        if self.df is None:
            self.load_dataset()

        X = self.df[self.features]
        y = self.df['price']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)
        self.r2 = r2_score(y_test, y_pred)
        self.is_trained = True

        self.display_r_squared()

    def display_r_squared(self):
        """Display the R-squared score of the trained model."""
        print("\n" + "=" * 60)
        print("           MODEL EVALUATION")
        print("=" * 60)
        print(f"  Features used : {self.features}")
        print(f"  R-squared     : {self.r2:.4f}")
        print("=" * 60)

    # ------------------------------------------------------------------ #
    #  Prediction
    # ------------------------------------------------------------------ #
    def predict_price(self):
        """Prompt user for feature values and predict house price."""
        if not self.is_trained:
            print("\n  [!] Model is not trained yet. Please train first.")
            return None

        print("\n" + "=" * 60)
        print("           PREDICT HOUSE PRICE")
        print("=" * 60)
        print("  Enter values for each feature:\n")

        values = {}
        for feat in self.features:
            while True:
                try:
                    val = float(input(f"    {feat}: "))
                    values[feat] = val
                    break
                except ValueError:
                    print("    Please enter a valid number.")

        input_df = pd.DataFrame([values])
        predicted = self.model.predict(input_df)[0]

        print("-" * 60)
        print(f"  Predicted House Price: ${predicted:,.2f}")
        print("=" * 60)

        self.save_prediction(values, predicted)
        return predicted

    # ------------------------------------------------------------------ #
    #  File handling — data.txt
    # ------------------------------------------------------------------ #
    def save_prediction(self, input_values, predicted_price):
        """Append a prediction record to data.txt."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        features_str = ", ".join(f"{k}={v}" for k, v in input_values.items())
        line = f"{timestamp} | {features_str} | Predicted: ${predicted_price:,.2f}\n"
        with open(self.data_file, 'a') as f:
            f.write(line)
        print(f"  [✓] Prediction saved to {self.data_file}")

    def load_predictions(self):
        """Read and display prediction history from data.txt."""
        print("\n" + "=" * 60)
        print("           PREDICTION HISTORY")
        print("=" * 60)
        try:
            with open(self.data_file, 'r') as f:
                lines = f.readlines()
            if not lines:
                print("  No predictions recorded yet.")
            else:
                for i, line in enumerate(lines, 1):
                    print(f"  {i}. {line.strip()}")
        except FileNotFoundError:
            print("  No predictions recorded yet.")
        print("=" * 60)
