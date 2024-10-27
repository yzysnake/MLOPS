import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import shutil
import subprocess
from git import Repo as GitRepo, GitCommandError
from dvc.repo import Repo as DvcRepo

def run_eda(df):
    """
    Runs Exploratory Data Analysis (EDA) on the provided dataset and generates visualizations.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    - None
    """
    # Display the number of null values in each column
    null_counts = df.isnull().sum()
    print("Number of null values in each column:")
    print(null_counts)
    print("\n")

    # Define categorical and numerical variables
    categorical_vars = ['region', 'gender', 'how long']
    numerical_vars = ['age', 'height', 'weight', 'fran', 'helen', 'grace',
                      'filthy50', 'fgonebad', 'run400', 'run5k', 'candj',
                      'snatch', 'deadlift', 'backsq', 'pullups']

    # Filter variables to those present in df
    available_categorical_vars = [var for var in categorical_vars if var in df.columns]
    available_numerical_vars = [var for var in numerical_vars if var in df.columns]

    # Set plot style for better visuals
    sns.set_style("whitegrid")

    # Categorical variables bar charts including NaN values
    for var in available_categorical_vars:
        plt.figure(figsize=(10, 6))
        # Replace NaN with 'NaN' string to include in counts
        df[var] = df[var].astype('object').fillna('NaN')
        # Calculate value counts including NaN
        value_counts = df[var].value_counts()
        # Create a DataFrame for plotting
        plot_data = pd.DataFrame({var: value_counts.index, 'Count': value_counts.values})
        # Plot using hue and set legend to False
        sns.barplot(data=plot_data, x=var, y='Count', hue=var, palette="viridis", dodge=False, legend=False)
        plt.title(f"Distribution of {var.capitalize()}", fontsize=14)
        plt.xlabel(var.capitalize(), fontsize=12)
        plt.ylabel("Count", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    # Generate a list of colors for numerical variables
    colors = sns.color_palette("coolwarm", n_colors=len(available_numerical_vars))

    # Numerical variables box plots ignoring NaN values
    for idx, var in enumerate(available_numerical_vars):
        # Ensure the variable is numeric
        df[var] = pd.to_numeric(df[var], errors='coerce')
        if df[var].dropna().empty:
            continue  # Skip if no valid data
        plt.figure(figsize=(6, 8))
        sns.boxplot(y=df[var], color=colors[idx])
        plt.title(f"Box Plot of {var.capitalize()}", fontsize=14)
        plt.ylabel(var.capitalize(), fontsize=12)
        plt.tight_layout()
        plt.show()

    # Correlation matrix for numerical variables
    if available_numerical_vars:
        # Select numerical columns that are present in df
        df_numeric = df[available_numerical_vars].select_dtypes(include=['float64', 'int64'])
        df_numeric = df_numeric.dropna(axis=0, how='all')

        if not df_numeric.empty:
            # Compute the correlation matrix
            corr_matrix = df_numeric.corr()

            # Plot the correlation heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Spectral", cbar=True)
            plt.title("Correlation Matrix of Numerical Variables", fontsize=16)
            plt.show()
        else:
            print("No numerical data available for correlation matrix.")
    else:
        print("No numerical variables available in the DataFrame for analysis.")



def train_linear_model(df):
    """
    Trains a linear regression model to predict 'total_lift' using 'age', 'height', and 'weight' as explanatory variables.
    Fills any null values with the median value of the respective columns.
    Splits the data into training and testing sets (80% train, 20% test) with random state 32.
    Shows the final metrics and a graph showing the model performance.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    - None
    """
    # Check if required columns are present
    required_columns = ['age', 'height', 'weight', 'Total Lift']
    available_columns = df.columns.tolist()
    missing_columns = [col for col in required_columns if col not in available_columns]

    if missing_columns:
        print(f"The following required columns are missing from the DataFrame: {missing_columns}")
        print("Cannot proceed with model training.")
        return

    # Select features and target variable
    X = df[['age', 'height', 'weight']]
    y = df['Total Lift']

    # Fill null values with median values
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # Ensure all data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop any remaining NaN values after conversion
    data = pd.concat([X, y], axis=1).dropna()
    X = data[['age', 'height', 'weight']]
    y = data['Total Lift']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=32
    )

    # Create and train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # Plotting actual vs predicted values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue', edgecolor='white', s=70)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Actual vs. Predicted Total Lift")
    plt.xlabel("Actual Total Lift")
    plt.ylabel("Predicted Total Lift")
    plt.tight_layout()
    plt.show()

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from opacus import PrivacyEngine
import matplotlib.pyplot as plt
import seaborn as sns

def train_dp_model_pytorch(df):
    """
    Trains a differentially private linear regression model using PyTorch and Opacus.
    Predicts 'total_lift' using 'age', 'height', and 'weight' as explanatory variables.
    Fills any null values with the median value of the respective columns.
    Splits the data into training and testing sets (80% train, 20% test) with random state 32.
    Shows the final metrics and a graph showing the model performance.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the dataset.

    Returns:
    - None
    """

    # Check if required columns are present
    required_columns = ['age', 'height', 'weight', 'Total Lift']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        print(f"The following required columns are missing from the DataFrame: {missing_columns}")
        print("Cannot proceed with model training.")
        return

    # Select features and target variable
    X = df[['age', 'height', 'weight']]
    y = df['Total Lift']

    # Fill null values with median values
    X = X.fillna(X.median())
    y = y.fillna(y.median())

    # Ensure all data is numeric
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')

    # Drop any remaining NaN values after conversion
    data = pd.concat([X, y], axis=1).dropna()
    X = data[['age', 'height', 'weight']]
    y = data['Total Lift']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=32
    )

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define the model
    model = nn.Linear(3, 1)  # 3 input features, 1 output

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # Privacy parameters
    noise_multiplier = 1.0  # Adjust based on desired privacy/accuracy trade-off
    max_grad_norm = 1.0  # Gradient clipping

    # Initialize the privacy engine
    privacy_engine = PrivacyEngine()

    # Attach the privacy engine to the optimizer and the model
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
    )

    # Training loop
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)

        epoch_loss /= len(train_loader.dataset)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    # Evaluate the model on the test set
    model.eval()
    with torch.no_grad():
        y_pred_tensor = model(X_test_tensor)
    y_pred = y_pred_tensor.numpy().flatten()

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nDifferentially Private Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"R-squared (R²): {r2:.2f}")

    # Plotting actual vs predicted values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='purple', edgecolor='white', s=70)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.title("Actual vs. Predicted Total Lift (DP Model)")
    plt.xlabel("Actual Total Lift")
    plt.ylabel("Predicted Total Lift")
    plt.tight_layout()
    plt.show()

    # Get the actual epsilon achieved
    epsilon_spent = privacy_engine.get_epsilon(delta=1e-5)
    print(f"Achieved Differential Privacy with ε = {epsilon_spent:.2f} and δ = 1e-5")




# Define Paths
GIT_REPO_DIR = '/mnt/c/Users/yzy_s/PycharmProjects/MLOPS/'
DVC_PROJECT_DIR = '/mnt/c/Users/yzy_s/PycharmProjects/MLOPS/Assignment_1/'

# Ensure paths end with '/'
if not GIT_REPO_DIR.endswith('/'):
    GIT_REPO_DIR += '/'
if not DVC_PROJECT_DIR.endswith('/'):
    DVC_PROJECT_DIR += '/'


# Initialize Git Repository
def initialize_git_repo():
    try:
        git_repo = GitRepo(GIT_REPO_DIR)
        print("Git repository already initialized.")
    except GitCommandError:
        git_repo = GitRepo.init(GIT_REPO_DIR)
        print("Initialized new Git repository.")
    return git_repo


# Initialize DVC Repository
def initialize_dvc_repo(git_repo):
    try:
        dvc_repo = DvcRepo(DVC_PROJECT_DIR)
        print("DVC repository already initialized.")
    except Exception:
        dvc_repo = DvcRepo.init(DVC_PROJECT_DIR, subdir=True, force=True)
        print("Initialized DVC repository in subdirectory.")
    return dvc_repo


# Add and Commit Dataset Version
def add_commit_dataset(git_repo, dvc_repo, version_tag, dataset_file):
    """
    Add a dataset to DVC, commit changes to Git, and tag the commit.

    Args:
        git_repo (GitRepo): The Git repository object.
        dvc_repo (DvcRepo): The DVC repository object.
        version_tag (str): The tag name for the version (e.g., 'v1').
        dataset_file (str): The CSV file to add (e.g., 'athletes_datav1.csv').
    """
    dataset_path = os.path.join(DVC_PROJECT_DIR, dataset_file)

    # Ensure dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset file {dataset_file} does not exist in {DVC_PROJECT_DIR}.")
        return

    # Add dataset to DVC
    dvc_repo.add(dataset_file)
    print(f"Added {dataset_file} to DVC tracking.")

    # Stage changes for Git (relative path from git repo)
    relative_path = os.path.relpath(os.path.join(DVC_PROJECT_DIR, f"{dataset_file}.dvc"), GIT_REPO_DIR)
    git_repo.index.add([relative_path, os.path.relpath(os.path.join(DVC_PROJECT_DIR, ".gitignore"), GIT_REPO_DIR)])

    # Commit changes
    commit_message = f"Add dataset version {version_tag}"
    git_repo.index.commit(commit_message)
    print(f"Committed {version_tag} to Git.")

    # Tag the commit
    git_repo.create_tag(version_tag)
    print(f"Tagged commit as {version_tag}.")


# Replace Dataset and Commit as New Version
def update_dataset(git_repo, dvc_repo, new_dataset_file, version_tag):
    """
    Replace the current dataset with a new version, update DVC, commit, and tag.

    Args:
        git_repo (GitRepo): The Git repository object.
        dvc_repo (DvcRepo): The DVC repository object.
        new_dataset_file (str): The new CSV file to replace the current one.
        version_tag (str): The tag name for the new version (e.g., 'v2').
    """
    current_dataset = 'athletes_datav1.csv'
    new_dataset_path = os.path.join(DVC_PROJECT_DIR, new_dataset_file)

    # Ensure new dataset exists
    if not os.path.exists(new_dataset_path):
        print(f"New dataset file {new_dataset_file} does not exist in {DVC_PROJECT_DIR}.")
        return

    # Remove the old dataset file
    old_dataset_path = os.path.join(DVC_PROJECT_DIR, current_dataset)
    if os.path.exists(old_dataset_path):
        os.remove(old_dataset_path)
        print(f"Removed old dataset {current_dataset}.")
    else:
        print(f"Old dataset {current_dataset} does not exist. Skipping removal.")

    # Rename new dataset to the original name
    os.rename(new_dataset_path, old_dataset_path)
    print(f"Renamed {new_dataset_file} to {current_dataset}.")

    # Update DVC tracking
    dvc_repo.add(current_dataset)
    print(f"Updated {current_dataset} in DVC tracking.")

    # Stage changes for Git
    relative_path = os.path.relpath(os.path.join(DVC_PROJECT_DIR, f"{current_dataset}.dvc"), GIT_REPO_DIR)
    git_repo.index.add([relative_path])

    # Commit changes
    commit_message = f"Update dataset to version {version_tag}"
    git_repo.index.commit(commit_message)
    print(f"Committed {version_tag} to Git.")

    # Tag the commit
    git_repo.create_tag(version_tag)
    print(f"Tagged commit as {version_tag}.")


# Checkout to a Specific Version
def checkout_version(git_repo, dvc_repo, tag):
    """
    Checkout a specific Git tag and update DVC data accordingly.

    Args:
        git_repo (GitRepo): The Git repository object.
        dvc_repo (DvcRepo): The DVC repository object.
        tag (str): The Git tag to checkout (e.g., 'v1' or 'v2').
    """
    try:
        # Checkout the specified Git tag
        git_repo.git.checkout(tag)
        print(f"Checked out to Git tag '{tag}'.")

        # Update DVC data
        dvc_repo.checkout()
        print(f"DVC data updated to version '{tag}'.")
    except GitCommandError as e:
        print(f"Error checking out to tag '{tag}': {e}")


# Load Data into pandas DataFrame
def load_data_into_dataframe(dataset_file):
    """
    Load the current version of the dataset into a pandas DataFrame.

    Args:
        dataset_file (str): The CSV file to load (e.g., 'athletes_datav1.csv').

    Returns:
        pd.DataFrame: The loaded DataFrame or None if an error occurs.
    """
    dataset_path = os.path.join(DVC_PROJECT_DIR, dataset_file)
    try:
        df = pd.read_csv(dataset_path)
        print(f"Loaded data from '{dataset_path}'.")
        return df
    except Exception as e:
        print(f"Error loading data from '{dataset_path}': {e}")
        return None


# List Available Git Tags
def list_git_tags(git_repo):
    """
    List all Git tags in the repository.

    Args:
        git_repo (GitRepo): The Git repository object.
    """
    tags = git_repo.tags
    print("Available Git Tags:")
    for tag in tags:
        print(f"- {tag}")