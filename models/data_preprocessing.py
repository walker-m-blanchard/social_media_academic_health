import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split as tts

SEED = 42

# Encodes nominal and ordinal data into numerical data
def encoding(data):
    # Changes into binary values
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['Affects_Academic_Performance'] = label_encoder.fit_transform(
        data['Affects_Academic_Performance'])

    academic_map = {'High School': 0, 'Undergraduate': 1, 'Graduate': 2}
    data['Academic_Level'] = data['Academic_Level'].map(academic_map)

    # Uses one-hot encoding for attributes with multiple nominal values
    categorical_cols = ['Most_Used_Platform', 'Relationship_Status']
    new_cols = pd.get_dummies(data, columns=categorical_cols)
    new_student_data = data.assign(**new_cols)
    new_student_data.drop(columns=['Country', 'Age', 'Most_Used_Platform', 'Relationship_Status'], inplace=True)

    new_student_data.to_csv('data/encoded_student_data.csv', index=False)
    return new_student_data

# Transforms encoded data into normalized and standardized data
def feature_scaling(data, scaling):
    columns = ['Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score',
                           'Conflicts_Over_Social_Media', 'Addicted_Score', 'Academic_Level']

    scaler = None
    if scaling == 'Normal':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    scaled_data = data.copy(deep=True)
    scaled_data[columns] = pd.DataFrame(scaler.fit_transform(data[columns]))

    if scaling == 'Normal':
        scaled_data.to_csv('data/normalized_student_data.csv', index=False)
    else:
        scaled_data.to_csv('data/standardized_student_data.csv', index=False)

    return scaled_data

# Transforms standardized data into PCA components
def pca(data):
    x_data = data.drop(columns=['Affects_Academic_Performance', 'Mental_Health_Score', 'Addicted_Score', 'Student_ID'])
    y_data = data['Affects_Academic_Performance']

    # All train-test splits in pipeline use 80/20 split and stratify based on academic impact of social media use
    x_train, x_test, y_train, y_test = tts(x_data, y_data, test_size=0.2, stratify=y_data, random_state = SEED)

    x_train_nparray = x_train.astype('float64')

    # Calculates eigenvalues
    cov_matrix = np.cov(x_train_nparray, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
    index = np.argsort(eig_vals)[::-1]
    eig_vals = eig_vals[index]

    # Prints eigenvalues for PCA evaluation
    plt.rcParams['font.size'] = 16
    fig, ax = plt.subplots(figsize=[12, 8], dpi=300)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(base=2.0))
    plt.plot(range(1, len(eig_vals) + 1), eig_vals, color='green', marker='o')
    plt.xlabel("Principal Component", fontsize=20)
    plt.ylabel("Eigenvalue", fontsize=20)
    plt.savefig('figures/scree_plot.png', dpi=300)

    # Trains PCA and transforms test data
    pca_model = PCA(n_components=2)
    pca_model.fit_transform(x_train)
    x_test_pca = pca_model.transform(x_test)

    # Calculates and prints loading matrix from PCA components
    loadings_std = pca_model.components_.T * np.sqrt(pca_model.explained_variance_)
    loading_matrix = pd.DataFrame(loadings_std, index=x_data.columns)
    loading_matrix.to_csv('data/loading_matrix.csv', index=False)

    # Stratifies data based on y_data for scatter plot coloring
    unaffected = x_test_pca[y_test == 0, :]
    affected = x_test_pca[y_test == 1, :]

    # Prints scatter plot of the test data on the two new PCA axis
    plt.figure(figsize=[12, 8], dpi=300)
    plt.scatter(unaffected[:, 0], unaffected[:, 1], label='Unaffected Academics', c='#800000')
    plt.scatter(affected[:, 0], affected[:, 1], label='Affected Academics', c='#6B8F4E')
    plt.xlabel('PCA Component 1', fontsize=20)
    plt.ylabel('PCA Component 2', fontsize=20)
    plt.legend()
    plt.savefig('figures/pca_test_plot.png', dpi=300)

    pca_transformed_x = pd.DataFrame(pca_model.transform(x_data))
    pca_transformed_x.to_csv('data/pca_transformed_x.csv', index=False)

    return pca_transformed_x

# This script is used to preprocess survey data to encode non-numeric data, scale data, and reduce components
# Work being done to convert script into module
def main(seed):
    random.seed(seed)
    student_data = pd.read_csv('data/student_data.csv')

    student_data = encoding(student_data)
    std_data = feature_scaling(student_data, 'Standard')
    nrm_data = feature_scaling(student_data, 'Normal')
    pca_data = pca(std_data)

    return student_data, std_data, nrm_data, pca_data

if __name__ == '__main__':
    main(SEED)