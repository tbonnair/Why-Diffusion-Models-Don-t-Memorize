## Codes for the theoretical section

This repository contains the codes to reproduce the figures of the theory section of  **"Why Diffusion Models Don't Memorize: The Role of Implicit Dynamical Regularization in Training"**  by *T. Bonnaire, R. Urfin, G. Biroli, and M. Mézard*.  It contains the following codes.

---

## plot_spectrum_U.py

The script solves the saddle point equations for the Stieltjes transform of the random matrix $U$, and compares the **theoretical spectral density** $\rho(\lambda)$ with **empirical histograms**. This code reproduces the plots of Fig. 4 of the article

The model assumes Gaussian data with covariance:

$$
\rho_\Sigma(\lambda) = 0.5\,\delta(\lambda - \lambda_1) + 0.5\,\delta(\lambda - \lambda_2)
$$

---

## Training_random_features.py

The script trains a Random features Neural Network either with gradient descent or Adam and computes the train and test losses as well as the L2 error to the true score which are plotted in Fig.5 of the article.
