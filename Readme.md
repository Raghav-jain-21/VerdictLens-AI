🏛️ VerdictLens — AI-Powered Court Case Outcome Prediction & Alimony Evaluation
Show Image
Show Image
Show Image
Show Image
Show Image

An AI system that predicts Indian court case outcomes across Divorce, Custody, Maintenance, and Other Family cases using ML on 50,000+ historical legal judgments — built for lawyers, citizens, and legal researchers.


📌 Problem Statement
Legal case outcomes in India are often uncertain and depend on hundreds of complex factors. There is no easily accessible tool that provides data-driven predictive insights for case results. VerdictLens solves this by applying state-of-the-art Machine Learning on real historical legal data from Dev Data Lab and the Indian Kanoon API.

🚀 Features

⚖️ Multi-Case Outcome Prediction — predicts outcomes for Divorce, Custody, Maintenance, and Other Family cases
📊 Feature Importance Analysis — identifies which factors most influence case outcomes
🔍 Explainable AI — transparent decision-making with feature importance scores
🌐 REST API — Flask API endpoints for real-time predictions
💻 Web Interface — user-friendly dashboard via index.html
☁️ Cloud Ready — deployable on AWS


📊 Model Performance (Actual Results)
Case TypeTop FeaturesAccuracyDivorcedist_name (0.384), court_no (0.324), year (0.292)77.8%Custodydist_name (0.550), court_no (0.450)68.4%Maintenancedist_name (1.0)61.3%Other Familydist_name (1.0)60.7%
Key Insights

🏆 Divorce prediction achieves the highest accuracy at 77.8%
📍 District name is the most important feature across all case types
⚖️ Court number and year are strong secondary predictors for divorce cases
👩 Female petitioner status has zero importance — showing gender-neutral predictions


🛠️ Tech Stack
CategoryTechnologiesLanguagePython 3.10ML ModelsXGBoost, Random Forest, Scikit-learnData ProcessingPandas, NumPyVisualizationMatplotlib, SeabornWeb FrameworkFlaskFrontendHTML5, CSS3, JavaScriptDatabaseMongoDBCloudAWSData SourceDev Data Lab, Indian Kanoon API

📂 Dataset

Source: Dev Data Lab — Indian court case records
Size: 1GB+ (50,000+ Indian legal judgments)
Cases covered: Divorce, Custody, Maintenance, Other Family
Download: Google Drive Dataset Link

Setup Instructions
bash# Download from Google Drive link above
# Then extract into the Output/dataset/ folder:
unzip justice_data.zip -d Output/dataset/

📁 Project Structure
VerdictLens-AI/
├── app.py                          ← Main Flask application
├── requirements.txt                ← Python dependencies
├── README.md                       ← You are here
│
├── models/
│   ├── alimony_calculator.json     ← Alimony calculation model
│   ├── divorce_categorical_model.pkl
│   ├── custody_categorical_model.pkl
│   ├── maintenance_categorical_model.pkl
│   ├── other_family_categorical_model.pkl
│   └── model_info.json             ← Model metadata
│
├── src/
│   ├── collecting_data.py          ← Data collection pipeline
│   ├── decode_02.py                ← Data decoding
│   ├── generate_report.py          ← Report generation
│   ├── inspect_keys.py             ← Data inspection utilities
│   ├── train_model_ali.py          ← Alimony model training
│   └── train_model.py              ← Main model training
│
├── Output/
│   ├── dataset/                    ← (not pushed — download from Drive)
│   │   ├── custody_cases.csv
│   │   ├── divorce_cases.csv
│   │   ├── maintenance_cases.csv
│   │   └── other_family_cases.csv
│   └── project_report_images/
│       ├── alimony_evaluation.png
│       ├── confusion_matrix.png
│       ├── feature_importance.png
│       └── prediction_model.png
│
├── Static/
│   ├── script.js
│   └── style.css
│
├── templates/
│   └── index.html
│
└── .gitignore

⚙️ How to Run Locally
1. Clone the repository
bashgit clone https://github.com/Raghav-jain-21/VerdictLens-AI.git
cd VerdictLens-AI
2. Create a virtual environment
bashpython -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
3. Install dependencies
bashpip install -r requirements.txt
4. Download the dataset
Download from Google Drive and extract into Output/dataset/
5. Train the models
bashpython src/train_model.py
6. Run the app
bashpython app.py
# Open http://localhost:5000 in your browser

🔮 Future Improvements

 Improve Maintenance and Other Family accuracy with advanced NLP features
 Add support for criminal and property case types
 Multilingual support (Hindi, regional languages)
 Deploy live demo on AWS EC2
 Add SHAP/LIME explainability for individual predictions


👨‍💻 Author
Raghav Jain

📧 jainraghav794@gmail.com
💼 LinkedIn
🐙 GitHub


📄 License
This project is licensed under the MIT License.

⭐ If you found this project useful, please give it a star!