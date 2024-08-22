% Clear the command window, close all figures, and clear all workspace variables
clc;
close all;
clear all;

% Define problem parameters
N = 20;         % Number of rows in design matrix
M = 40;         % Number of columns in design matrix
D = 7;          % Number of non-zero coefficients in weight vector

% Define noise levels in decibels (dB)
noise_dB = [-20 -15 -10 -5 0];

% Convert noise levels from dB to linear scale (variance)
noise_variance = 10.^(noise_dB / 10);

% Initialize array to store average NMSE (Normalized Mean Squared Error)
NMSE_avg = zeros(length(noise_variance), 1);

% Loop over different noise levels
for k = 1:length(noise_variance)
    NMSE = 0;
    % Perform iterations for each noise level
    for j = 1:100

        % Generate random design matrix (phi)
        phi = randn(N, M);

        % Initialize weight vector with D non-zero coefficients
        w = zeros(M, 1);
        idx = randperm(M, D);
        w(idx) = randn(D, 1);

        % Generate noise vector with specified variance
        epsilon = normrnd(0, sqrt(noise_variance(k)), [N, 1]);

        % Generate target vector (t) using design matrix, weight vector, and noise
        t = phi * w + epsilon;

        % Initialize precision parameter (alpha)
        alpha = repmat(100, M, 1);

        % Initialize precision matrix (A) as diagonal matrix with alpha values
        A = diag(alpha);

        % Initialize posterior covariance matrix (sigma) and mean vector (mu)
        sigma = inv(noise_variance(k)^(-1) * (phi' * phi) + A);
        mu = (noise_variance(k)^(-1)) * (sigma * phi' * t);

        % Iteratively update alpha, sigma, and mu using sparse Bayesian learning
        mu_new = zeros(M, 1);
        i = 1;
        while (i < 500)
            for l = 1:length(alpha)
                gamma = 1 - (alpha(l) * sigma(l, l));
                alpha(l) = gamma / (mu(l)^.2);
            end
            A = diag(alpha);
            sigma = inv(noise_variance(k)^(-1) * (phi' * phi) + A);
            mu_new = (noise_variance(k)^(-1)) * (sigma * phi' * t);
            % Check convergence criterion
            if ((norm(mu_new - mu) / norm(mu))^2 <= 0.001)
                break;
            end
            i = i + 1;
        end

        % Compute NMSE and accumulate for averaging
        NMSE = NMSE + (norm(mu_new - w) / norm(w))^2;
    end
    % Compute average NMSE for current noise level
    NMSE_avg(k) = NMSE / 100;
end

% Plot the results
plot(noise_dB, NMSE_avg, 'LineWidth', 1);
hold on
scatter(noise_dB, NMSE_avg, 'filled', 'Marker', 'o');
hold off
xlabel("Noise in dB")
ylabel("Average Noise MSE")
title("Sparse Bayesian Learning")
