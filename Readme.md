# VERITAS: Verifiable Evaluation and Reporting In Transparent AI Systems

**Abstract:**  
VERITAS is a self-introspective engine designed to automatically generate comprehensive model cards for machine learning systems. By integrating performance evaluation, explainability via SHAP, preliminary bias auditing, and governance/compliance logging, VERITAS enhances transparency and accountability in AI deployments. This repository contains a proof-of-concept implementation using the Iris dataset and a RandomForestClassifier, demonstrating how a model can autonomously document its capabilities, limitations, and operational metrics.

**Author:**  
Gerald Enrique Nelson Mc Kenzie


## Overview

VERITAS aims to simplify the process of generating model cards by automating key aspects of model evaluation and reporting. This project includes:
- **Performance Evaluation:** Automatically computes metrics such as accuracy and confusion matrix.
- **Explainability:** Uses SHAP to provide feature-level insights for model predictions.
- **Bias Audit:** Conducts a preliminary bias audit (placeholder) to highlight fairness considerations.
- **Governance & Compliance:** Logs audit information and embeds compliance details within the report.

## Repository Structure

- **`veritas_monitor.py`**: Contains the main implementation of the `VeritasMonitor` class that wraps a trained model and generates a model card.
- **`requirements.txt`**: Lists the Python dependencies required to run the code.
- **`README.md`**: This file, which provides an overview and usage instructions.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/lordxmen2k/VERITAS.git
   cd VERITAS

2. **Create a Virtual Environment (Optional but Recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate   # On Windows use: venv\Scripts\activate

3. **Install Dependencies:**
    ```bash
    pip install -r requirements.txt

## Usage

1. VERITAS comes with an example that uses the Iris dataset and a RandomForestClassifier. To run the example:
   ```bash
    python veritas_monitor.py

2.
Upon execution, the script will:

    Load the Iris dataset and split it into training and testing sets.
    Train a RandomForestClassifier on the training data.
    Evaluate the model's performance and generate SHAP explanations for a selected test sample.
    Compile a comprehensive model card (in JSON format) that includes performance metrics, explainability data, bias audit results, and compliance logs.
    Save the model card as veritas_model_card.json in the repository directory.

## Extending VERITAS

    Using Your Own Data/Model: Replace the Iris dataset and the RandomForestClassifier with your dataset and model. Ensure that your data is appropriately preprocessed and that the model supports the methods used in VERITAS (e.g., predict, get_params).
    Enhancing Bias Audits: The current bias audit is a replaceable. You can integrate more sophisticated fairness metrics and tests based on your project's needs.
    Integrating Additional Logging: Expand the audit_logs functionality to include more detailed logs, such as secure logging or integration with a blockchain for audit trails.    

## License & Citation

    This project is open-source under the MIT License. If you use this framework in your research, please cite:

@article{Veritas2024,
  author = {Gerald Enrique Nelson Mc Kenzie},
  title = {VERITAS: Verifiable Evaluation and Reporting In Transparent AI Systems },
  journal = {Zenodo},
  year = {2024},
  doi = {10.5281/zenodo.14811299},
  url = {https://github.com/lordxmen2k/Veritas}
}
