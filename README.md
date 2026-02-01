# Severity-MLGenetic Disease Severity Prediction Pipeline

This repository contains a machine learning pipeline designed to predict the severity of genetic diseases (Mild, Moderate, Severe, Profound). It integrates mechanistic biological features with phenotypic data to train robust predictive models.

The pipeline utilizes LightGBM (Gradient Boosting) and PyTorch (Deep Neural Networks) for classification, providing comprehensive metrics, SHAP-based interpretability, and automated LLM-generated explanations for predictions.

🚀 Key Features

Multi-Scenario Analysis: Evaluates performance across different feature sets:

S1: Mechanistic Only (Biological/Gene features)

S5: Phenotype Only (Clinical features)

S2: Combined (Mechanistic + Phenotype)

S3/S4: Integration with Variant-level data.

Dual Modeling Approach: Compares LightGBM (Tree-based) and Deep Neural Networks (PyTorch).

Explainable AI (XAI):

SHAP Analysis: Identifies top features driving severity predictions.

LLM Integration: Connects to OpenRouter (e.g., GPT-4o) to generate natural language explanations for clinical context.

Robust Evaluation: Outputs ROC curves, Precision-Recall metrics, F1 scores, and MCC.

📂 Project Structure

.
├── config.py              # Central configuration (paths, params, API keys)
├── main.py                # Main execution script
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (API keys) - DO NOT COMMIT
├── .gitignore             # Git ignore rules
├── data/                  # Input data directory (CSV files)
│   ├── final_output_with_severity.csv
│   ├── CORE_MECHANISTIC_FEATURES.csv
│   └── DOMAIN_CARRIER_FEATURES.csv
└── output/                # Generated results
    ├── figures/           # ROC curves and plots
    ├── results/           # Metrics CSVs and raw SHAP data
    ├── models/            # Saved model files
    └── explanations/      # LLM generated text


🛠️ Setup & Installation

1. Clone the Repository

git clone [https://github.com/yourusername/severity-prediction.git](https://github.com/yourusername/severity-prediction.git)
cd severity-prediction


2. Install Dependencies

It is recommended to use a virtual environment.

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


3. Configure Environment

Create a .env file in the root directory to store your API keys safely.

cp .env.example .env


Edit .env and add your OpenRouter API key if you plan to use the LLM explanation feature:

OPENROUTER_API_KEY=your_actual_key_here


4. Prepare Data

Place your dataset files in the data/ folder. The pipeline expects the following files (paths defined in config.py):

final_output_with_severity.csv (Primary phenotype/disease data)

CORE_MECHANISTIC_FEATURES.csv (Gene-level biological features)

DOMAIN_CARRIER_FEATURES.csv (Optional variant features)

🏃 Usage

Run the main pipeline script:

python main.py


What happens when you run it?

Data Loading: Merges phenotype, mechanistic, and variant data.

Preprocessing: Handles missing values, scaling, and leakage prevention.

Training: Trains LightGBM and DNN models on multiple feature scenarios (S1-S5).

Evaluation: Saves metrics (AUC, F1, MCC) to output/results/metrics.csv.

Visualization: Generates ROC curves in output/figures/.

Explanation: If enabled, calculates SHAP values and queries the LLM for explanations.

📊 Outputs

Check the output/ directory for results:

pipeline_log.txt: Detailed execution log.

metrics.csv: Comparative performance table of all models/scenarios.

figures/roc_*.png: ROC curves for each scenario.

⚙️ Configuration

You can adjust parameters in config.py:

TEST_SIZE: Fraction of data used for testing (Default: 0.2).

RANDOM_SEED: Ensure reproducibility.

LLM_MODEL: Change the model used for explanations (e.g., openai/gpt-4o).

TARGET_MAP: Adjust severity class mappings.

🤝 Contributing

Fork the repository.

Create your feature branch (git checkout -b feature/NewFeature).

Commit your changes.

Push to the branch and open a Pull Request.

📜 License

MIT License
