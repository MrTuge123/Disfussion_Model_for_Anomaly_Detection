Big picture:

Digit 0 is treated as normal and digits 1–9 are treated as anomalies.

We train a neural network to learn how to nudge noisy images back toward what normal zeros look like (effectively calculating the score). The size of this nudge, measured by the norm of the score function, is used as an anomaly score. 


Part A: How the score function is computed and learned

1. Definition of the score function
The score function is a neural network that takes as input a noisy MNIST image and a noise level. It outputs a tensor of the same shape as the image (28×28). Each output value represents how that pixel should change to make the image look more like a typical zero.

2. Training data selection
From the MNIST training set, keep only images with label 0. Discard all other digits during training. This ensures the model learns only the distribution of normal data.

3. Noise levels
Define a range of noise strengths (sigma values), for example between 0.01 and 0.3. During training, randomly sample a sigma value for each image. This exposes the model to different noise intensities.

4. Noise corruption
For each clean image x:
- Sample random Gaussian noise epsilon with the same shape as x.
- Sample a noise strength sigma.
- Create a noisy image: x_tilde = x + sigma * epsilon.

5. Training objective (conceptual)
The network is trained to predict how to move a noisy image back toward the clean data distribution. Since the added noise is known, the model learns a vector field that points toward regions of higher probability under the zero-digit distribution. After training, this network serves as the score function.


Part B: How anomaly scores are computed

6. Multiple noise evaluations per test image
For a test image x (any digit), evaluate it at multiple predefined noise levels. For example, use K = 4 noise levels such as {0.01, 0.03, 0.1, 0.3}.

For each noise level sigma_k:
- Sample Gaussian noise epsilon_k.
- Form a noisy image: x_tilde_k = x + sigma_k * epsilon_k.
- Compute the score output g_k = s_theta(x_tilde_k, sigma_k).
- Compute the score norm n_k = ||g_k||_2.

7. Aggregation into a single anomaly score
Average the score norms across noise levels:
A(x) = (1 / K) * sum_k n_k.

Interpretation:
- Small A(x): the image already looks like a normal zero and requires little correction.
- Large A(x): the image does not resemble a zero and requires large corrections.


Part C: Making anomaly decisions

8. Score computation on test data
Compute A(x) for every image in the MNIST test set. Assign ground truth labels:
- Label 0: normal
- Labels 1–9: anomaly

9. Thresholding
Choose a threshold T such that:
- If A(x) > T, classify the image as an anomaly.
- Otherwise, classify it as normal.

The threshold can be chosen by inspecting score histograms or by fixing an acceptable false positive rate on digit 0.

Notes
This plan is intentionally simple and designed for fast iteration. More advanced extensions can include trajectory-based scores, likelihood comparisons, or multi-class training, but those are deferred until this baseline is validated.

