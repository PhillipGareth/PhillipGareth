Healthcare GNN Project
A Graph Neural Network (GNN) project for predicting patient diseases based on healthcare data, implemented in Google Colab.
Project Overview
This project develops a Graph Neural Network (GNN) to predict diseases for patients using healthcare data, including patient demographics, medications, and medical conditions. The model leverages graph-based representations to capture relationships between patients and their medical histories, enabling accurate multi-label disease predictions. The primary goal is to identify which patients are likely to have specific diseases, outputting results with patient names, dates of birth, and predicted diseases (e.g., "Type 2 diabetes mellitus, Hypertension").
The project is implemented in a Google Colab notebook (GNN_Project.ipynb), which includes the following steps:

Data Preprocessing: Loads and processes patients.csv, conditions.csv, and medications.csv to create a feature-rich dataset.
Graph Creation: Constructs a graph where nodes represent patients and edges capture shared medical attributes.
Task Grouping: Organizes disease labels into groups for efficient GNN training.
GNN Training: Trains a GroupedGNN model using PyTorch Geometric to predict diseases.
Testing: Evaluates the model on a test set, computing metrics like accuracy and F1-score.
Prediction: Generates predictions for test patients, mapping disease codes to readable names.

The final output is a CSV file (patient_disease_predictions_detailed.csv) listing patient names, dates of birth, patient IDs, and their predicted diseases, suitable for inclusion in a project report or further analysis.
Dataset
The project uses three healthcare datasets stored in Google Drive:

patients.csv: Contains patient details such as ID, first name, last name, date of birth, gender, and other demographics.
conditions.csv: Records medical conditions with patient IDs, condition codes (e.g., SNOMED CT), and descriptions (e.g., "Chronic obstructive lung disease").
medications.csv: Lists medications prescribed to patients, including patient IDs and medication codes.

Note: Due to the sensitive nature of healthcare data, these datasets are not included in the repository. Users must provide their own datasets with the expected structure (see Data Requirements below).
The preprocessing step generates patient_data_sub.pkl, a processed dataset with patient features (e.g., age, gender, medication history) and disease labels, used for graph construction and model training.
Requirements
To run the project, you need:

A Google account with access to Google Colab and Google Drive.
Python 3.7 or higher.
The following Python packages (listed in requirements.txt):pandas
torch
torch-geometric
numpy
scikit-learn



Install dependencies in Colab by running:
!pip install -r requirements.txt

Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/Healthcare-GNN-Project.git

Replace your-username with your GitHub username.

Upload Datasets:

Place patients.csv, conditions.csv, and medications.csv in your Google Drive under /My Drive/.
Ensure the datasets have the required columns:
patients.csv: Id, FIRST, LAST, BIRTHDATE, GENDER, etc.
conditions.csv: PATIENT, CODE, DESCRIPTION, etc.
medications.csv: PATIENT, CODE, etc.


If using different paths, update the notebook’s data_dir variable.


Open the Notebook:

Go to Google Colab.
Click File > Upload notebook and select GNN_Project.ipynb.
Alternatively, open it directly from Google Drive if saved there.


Mount Google Drive:

Run the first cell to mount your Drive:from google.colab import drive
drive.mount('/content/drive')




Run the Notebook:

Execute all cells sequentially.
The notebook handles data loading, preprocessing, graph creation, model training, testing, and prediction.
Monitor outputs for errors (e.g., missing files or column mismatches).



Expected Outputs
Running GNN_Project.ipynb generates files in /content/drive/My Drive/GNN_Project/:

patient_data_sub.pkl: Processed dataset from Step 1.
gnn_model.pth: Trained GNN model from Step 5.
test_metrics.csv: Model performance metrics (e.g., accuracy, F1-score).
patient_disease_predictions_detailed.csv: Final predictions with columns:PATIENT_NAME,DATE_OF_BIRTH,PATIENT_ID,PREDICTED_DISEASES
"John Doe","1980-01-01","P005","Type 2 diabetes mellitus, Hypertension"
...



The predictions CSV is ideal for inclusion in a project report, summarizing which test patients are predicted to have specific diseases.
Data Requirements
Since the datasets are not included, users must provide their own with the following structure:

patients.csv:
Id: Unique patient identifier (renamed to PATIENT in code).
FIRST, LAST: Patient’s first and last names.
BIRTHDATE: Date of birth (e.g., “1980-01-01”).
Optional: GENDER, other demographics.


conditions.csv:
PATIENT: Patient identifier matching patients.csv’s Id.
CODE: Disease code (e.g., SNOMED CT 73595000).
DESCRIPTION: Disease name (e.g., “Chronic obstructive lung disease”).


medications.csv:
PATIENT: Patient identifier.
CODE: Medication code.



The notebook assumes these files are in /content/drive/My Drive/. Adjust paths in the code if stored elsewhere.
Notes

Sensitive Data: Do not share patients.csv, conditions.csv, medications.csv, patient_data_sub.pkl, gnn_model.pth, or patient_disease_predictions_detailed.csv publicly, as they may contain protected health information (PHI). The .gitignore file excludes these to prevent accidental commits.
Reproducibility: Results depend on the dataset provided. The notebook includes debugging prints to help diagnose issues like missing columns or code mismatches.
Performance: The model’s accuracy depends on the dataset size and quality. The GroupedGNN uses task grouping to handle multiple diseases, but group imbalances (e.g., [46, 1, 1, 1, 1]) may affect predictions for rare diseases.
Customizations: To modify features (e.g., add ethnicity) or labels, edit Step 1 in the notebook. For different disease codes (e.g., ICD-10 instead of SNOMED CT), update the code mapping logic.
GitHub: This repository contains only the notebook and supporting files (requirements.txt, .gitignore). Clone and run in Colab to replicate the project.

Troubleshooting

FileNotFoundError: Ensure datasets are in /content/drive/My Drive/. Check paths with:!ls "/content/drive/My Drive/"


KeyError or ValueError: Verify dataset columns match the expected structure. Print column names:print(pd.read_csv('/content/drive/My Drive/patients.csv').columns)


IndexError: Check if test_indices align with patient_data_sub rows. See notebook comments for fixes.
Non-Disease Predictions: Some SNOMED CT codes (e.g., “Only child”) may appear as labels. Filter them in Step 1 or update non_disease_codes in the prediction step.

For further issues, open a GitHub issue or check the notebook’s debug outputs.
Acknowledgments

Google Colab: For providing a free cloud-based environment to run the notebook.
PyTorch Geometric: For graph neural network implementation.
xAI: For support and inspiration in building AI-driven solutions.
Healthcare Data: The project assumes synthetic or anonymized data compliant with privacy regulations (e.g., HIPAA).

License
This project is licensed under the MIT License. See the LICENSE file for details (if included).

Feel free to star ⭐ this repository if you find it useful! For questions or contributions, please open an issue or submit a pull request.








# Healthcare GNN Project
Predicts patient diseases using a GNN.

## Usage
Run `GNN_Project.ipynb` in Colab with `patients.csv`, `conditions.csv`, `medications.csv` in Google Drive.

## Output
`patient_disease_predictions_detailed.csv`: Patient names, DOB, and predicted diseases.

## Notes
- Datasets not included.
- Requires `gnn_model.pth` from Step 5.
