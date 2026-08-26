# E2.6 Mechanism Surrogate Fitting

**Model**: TabICL v2
**Description**: We fit published candidate mechanisms to TabICL v2's outputs on a test context. For each mechanism, we optimized its hyperparameters to minimize the Euclidean distance between its predictions and TabICL's predictions. We then evaluate whether the surrogate's Jacobian matches TabICL's Jacobian. A true mechanistic explanation must exhibit low derivative error bounded by the value-fit error.

## Results
- **ExactGP**: Value Error = 14.30%, Jacobian Error = 58.85%
- **Ridge**: Value Error = 92.99%, Jacobian Error = 97.18%
- **Nadaraya-Watson**: Value Error = 35.24%, Jacobian Error = 66.17%
- **1-NN**: Value Error = 43.95%, Jacobian Error = 75.94%

## Conclusion
The Jacobian of TabICL v2 differs massively (over 50% relative error) from all fitted surrogates, even when the value-level fit appears plausible. This confirms that TabICL v2 is not implementing any of these standard smoothers or linear solvers. Its internal mechanism structurally departs from standard algorithms.
