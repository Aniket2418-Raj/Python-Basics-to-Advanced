import numpy as np
import matplotlib.pyplot as plt

class BayesianDiagnosis:
    """
    Bayesian inference for disease diagnosis with multiple independent tests.
    """
    def __init__(self, prior):
        """
        Initialize with prior probability of disease.
        """
        self.prior = prior
        self.posterior_history = [prior]

    def update(self, tests, results):
        """
        Update posterior probability given new test results.
        
        Parameters:
        - tests: list of tuples [(P(+|D), P(+|~D)), ...]
        - results: list of booleans [True/False, ...] representing positive/negative
        
        Returns:
        - posterior probability after all updates
        """
        P_D = self.posterior_history[-1]
        P_notD = 1 - P_D
        
        likelihood_given_D = 1
        likelihood_given_notD = 1
        
        for (P_pos_given_D, P_pos_given_notD), res in zip(tests, results):
            if res:  # Positive test
                likelihood_given_D *= P_pos_given_D
                likelihood_given_notD *= P_pos_given_notD
            else:    # Negative test
                likelihood_given_D *= 1 - P_pos_given_D
                likelihood_given_notD *= 1 - P_pos_given_notD
        
        P_all = P_D * likelihood_given_D + P_notD * likelihood_given_notD
        posterior = (P_D * likelihood_given_D) / P_all
        self.posterior_history.append(posterior)
        return posterior

    def plot_convergence(self):
        """
        Plot posterior probability convergence over updates.
        """
        plt.figure(figsize=(8,5))
        plt.plot(range(len(self.posterior_history)), self.posterior_history, 'o-', color='purple')
        plt.xlabel('Number of updates / tests', fontsize=12)
        plt.ylabel('Posterior Probability P(D|data)', fontsize=12)
        plt.title('Bayesian Posterior Probability Convergence', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.ylim(0,1)
        plt.show()



# Example Usage

# Prior probability of disease
prior = 0.001

# Define tests: (Sensitivity, False Positive Rate)
tests = [
    (0.95, 0.02),  # Test A
    (0.90, 0.05),  # Test B
    (0.92, 0.03)   # Test C
]

# Results for each test (True=positive, False=negative)
results = [True, True, False]

# Initialize BayesianDiagnosis
diagnosis = BayesianDiagnosis(prior)

# Update posterior step by step
for i in range(len(tests)):
    posterior = diagnosis.update([tests[i]], [results[i]])
    print(f"After Test {i+1} ({'Positive' if results[i] else 'Negative'}): Posterior = {posterior:.4f}")

# Plot convergence
diagnosis.plot_convergence()
