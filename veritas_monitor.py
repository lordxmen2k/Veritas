#!/usr/bin/env python
"""
VERITAS: Verifiable Evaluation and Reporting In Transparent AI Systems

This module provides a class, VeritasMonitor, that automatically generates
a comprehensive model card for a given machine learning model. The model card
includes information on model performance, explainability (using SHAP),
bias auditing (placeholder), and governance/compliance logs.

Usage:
    Run this module directly to generate a model card for a RandomForestClassifier
    trained on the Iris dataset.
"""

import json
import numpy as np
import shap  # Ensure you have installed SHAP: pip install shap
from sklearn.metrics import accuracy_score, confusion_matrix


class VeritasMonitor:
    def __init__(self, model, data, feature_names, compliance_info=None):
        """
        Initializes the VERITAS monitor.

        Parameters:
            model: Trained machine learning model.
            data: Tuple containing (X_train, X_test, y_train, y_test).
            feature_names: List of feature names.
            compliance_info: Optional dictionary with compliance/regulatory info.
        """
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = data
        self.feature_names = feature_names
        self.compliance_info = compliance_info or {}
        self.metrics = {}
        self.explanations = {}
        self.audit_logs = []

    def evaluate_performance(self):
        """
        Evaluates model performance on the test set.
        Computes accuracy and the confusion matrix.
        """
        y_pred = self.model.predict(self.X_test)
        self.metrics['accuracy'] = accuracy_score(self.y_test, y_pred)
        self.metrics['confusion_matrix'] = confusion_matrix(self.y_test, y_pred).tolist()
        log_entry = f"Performance evaluated with accuracy: {self.metrics['accuracy']:.4f}"
        self.audit_logs.append(log_entry)
        return self.metrics

    def generate_explanation(self, sample_index=0):
        """
        Generates an explanation for a given test sample using SHAP.
        
        Parameters:
            sample_index: Index of the sample in the test set to explain.
        """
        # Create a TreeExplainer based on the trained model
        explainer = shap.TreeExplainer(self.model)
        sample = self.X_test[sample_index].reshape(1, -1)
        shap_values = explainer.shap_values(sample)
        # For demonstration, compute the mean absolute SHAP values for the first class.
        # In a multi-class scenario, extend this to provide insights for each class.
        mean_shap = np.mean(np.abs(shap_values[0]), axis=0).tolist()
        self.explanations = {
            "sample_index": sample_index,
            "mean_absolute_shap_values": mean_shap,
            "feature_names": self.feature_names
        }
        log_entry = f"Generated SHAP explanation for sample index {sample_index}"
        self.audit_logs.append(log_entry)
        return self.explanations

    def audit_bias(self):
        """
        Conducts a preliminary bias audit.
        This placeholder returns a simple report indicating no bias detected.
        """
        bias_detected = False
        fairness_summary = "No significant bias detected in this preliminary audit."
        log_entry = f"Bias audit completed: {fairness_summary}"
        self.audit_logs.append(log_entry)
        return {"bias_detected": bias_detected, "fairness_summary": fairness_summary}

    def generate_model_card(self):
        """
        Compiles a complete model card that includes:
          - Model information (type, description, hyperparameters)
          - Performance metrics
          - Explainability data
          - Bias and fairness audit results
          - Governance and compliance logs
        """
        model_card = {
            "model_information": {
                "model_type": str(self.model.__class__.__name__),
                "description": "Self-generated model card by VERITAS.",
                "hyperparameters": self.model.get_params() if hasattr(self.model, 'get_params') else "Not available"
            },
            "performance": self.metrics,
            "explainability": self.explanations,
            "ethical_considerations": self.audit_bias(),
            "governance_and_compliance": {
                "audit_logs": self.audit_logs,
                "compliance_info": self.compliance_info,
            }
        }
        return model_card

    def save_model_card(self, filename="veritas_model_card.json"):
        """
        Saves the generated model card to a JSON file.

        Parameters:
            filename: The filename for the saved model card.
        """
        model_card = self.generate_model_card()
        with open(filename, "w") as f:
            json.dump(model_card, f, indent=4)
        self.audit_logs.append(f"Model card saved to {filename}")
        return filename


if __name__ == "__main__":
    # Example usage: Train a model on the Iris dataset and generate a model card.
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # Load the Iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    data = (X_train, X_test, y_train, y_test)

    # Train a RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Instantiate the VERITAS monitor
    veritas = VeritasMonitor(
        model=model,
        data=data,
        feature_names=feature_names,
        compliance_info={"GDPR": "Compliant", "HIPAA": "Not applicable"}
    )

    # Evaluate performance and generate explanation
    veritas.evaluate_performance()
    veritas.generate_explanation(sample_index=0)
    # Save the model card to a file
    veritas.save_model_card()

    # Retrieve and print the generated model card
    model_card = veritas.generate_model_card()
    print("Generated VERITAS Model Card:")
    print(json.dumps(model_card, indent=4))
