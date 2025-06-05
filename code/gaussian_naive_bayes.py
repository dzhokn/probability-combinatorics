import numpy as np
from dataclasses import dataclass

@dataclass
class GaussianNaiveBayes:
    features: np.ndarray
    labels: np.ndarray

    def fit(self) -> None:
        """Fits the Gaussian Naive Bayes model"""
        self.unique_labels = np.unique(self.labels)

        """
        self.features:
        [
            [1, 2, 3] label0
            [4, 5, 6] label1  # this will be filtered on first iteration
            [7, 8, 9] label0
        ]

        label_features.T:
        [[1, 7], [2, 8], [3, 9]]

        self.params:
        {
            (4, 7), # label0 mean and variance
            (5, 8)  # label1 mean and variance
        }
        """

        self.params = []
        for label in self.unique_labels:
            label_features = self.features[self.labels == label] # Filter features only for the current label
            self.params.append([(col.mean(), col.var()) for col in label_features.T]) # Calculate mean and variance for each feature

    def likelihood(self, data: float, mean: float, variance: float) -> float:
        """ Calculates the Gaussian likelihood of the data with given mean and variance"""
        eps = 1e-4 # Added in denominator to prevent division by zero

        coeff = 1 / np.sqrt(2 * np.pi * variance + eps)                 # coefficient of the Gaussian distribution
        exponent = np.exp(-(data - mean) ** 2 / (2 * variance + eps))   # exponent of the Gaussian distribution

        return coeff * exponent
    
    def predict(self,  features: np.ndarray) -> np.ndarray:
        """Performns inference using Bayes' theorem: P(A|B) = P(B|A) * P(A) / P(B)."""
        num_samples, _ = features.shape
        predictions = np.empty(num_samples)

        for idx, feature in enumerate(features):
            posteriors = []
            for label_idx, label in enumerate(self.unique_labels):
                # The mean of what we have
                prior = (self.labels == label).sum() / len(self.labels)

                # Likelihood - we will make a NAIVE assumption that each feature is independent:
                # P(a1, a2, a3 | B) = P(a1 | B) * P(a2 | B) * P(a3 | B)
                likelihood = np.prod(
                    [
                        self.likelihood(ft, m, v) for ft, (m, v) in zip(feature, self.params[label_idx])
                        ]
                )

                posterior = prior * likelihood # Posterior probability is calculated by multiplying the prior and likelihood
                posteriors.append(posterior)
            
            predictions[idx] = self.unique_labels[np.argmax(posteriors)] # Select the label with the highest posterior probability

        return predictions
    
### Testing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

features, labels = load_iris(return_X_y=True)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.5, random_state=42)
    

gnb = GaussianNaiveBayes(train_features, train_labels)
gnb.fit()

predictions = gnb.predict(test_features)

accuracy = np.mean(predictions == test_labels)
precision, recall, fscore, _ = precision_recall_fscore_support(test_labels, predictions, average='macro')

print(f"Accuracy: {round(accuracy, 3)}")
print(f"Precision: {round(precision, 3)}")
print(f"Recall: {round(recall, 3)}")
print(f"F-score: {round(fscore, 3)}")
