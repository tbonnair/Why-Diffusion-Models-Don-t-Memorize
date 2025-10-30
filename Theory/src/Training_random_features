#!/usr/bin/env python3
"""
Training script for the random feature diffusion model.

Implements gradient descent (SGD or Adam) dynamics on the learnable matrix A,
and tracks three types of losses:
    1. Train loss 
    2. Test loss 
    3. Error on the score 
"""

import os
import numpy as np
import torch
import torch.optim as optim
import argparse


# ======================================================
#                   Argument Parser
# ======================================================
def get_args():
    parser = argparse.ArgumentParser(
    )
    parser.add_argument("--psi_p", type=float, default=2.0, help="ψ_p = p/d")
    parser.add_argument("--psi_n", type=float, default=1.0, help="ψ_n = n/d")
    parser.add_argument("--d", type=int, default=int(2e3), help="data dimension")
    parser.add_argument("--t", type=float, default=1e-2, help="Diffusion time")
    parser.add_argument("--optimizer", choices=["GD", "Adam"], default="GD", help="Optimizer type")
    parser.add_argument("--num_epochs", type=int, default=int(5e6), help="Number of training iterations")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    return parser.parse_args()


# ======================================================
#            Model Definition
# ======================================================
def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Derived parameters
    Psi_p, Psi_n, d, t = args.psi_p, args.psi_n, args.d, args.t
    a_t = np.exp(-t)
    delta_t = 1 - a_t**2
    p, n = int(Psi_p * d), int(Psi_n * d)

    learning_rate = (1e-2) * d / delta_t # We use this scaling because the gradient is very small at initialization

    print(f"\n🚀 Training random feature diffusion model")
    print(f"Device: {device}")
    print(f"Parameters: ψ_p={Psi_p}, ψ_n={Psi_n}, d={d}, t={t}")
    print(f"Optimizer: {args.optimizer},lr={learning_rate:.2e}")
    print(f"Number of epochs: {args.num_epochs:,}\n")

    # Model components
    W = torch.randn(p, d, device=device)
    A = torch.zeros(d, p, requires_grad=True, device=device)
    x = torch.randn(n, d, device=device)

    # ------------------------------------------------------
    # Define S_A(x), here we use tanh as an activation function but it can be changed
    def model(x_input):
        return (A @ torch.tanh(W @ x_input.T / torch.sqrt(torch.tensor(d, device=device)))).T

    # ------------------------------------------------------
    # Loss function 
    def train_objective(A, x, eta):
        S_A_x = model(a_t * x + np.sqrt(delta_t) * eta)
        l2_term = torch.sum((np.sqrt(delta_t) * S_A_x + eta) ** 2) / (d * n)
        return l2_term 

    # ------------------------------------------------------
    # Train loss 
    def compute_train_loss(A, x, m=100):
        total_loss = 0
        with torch.no_grad():
            for _ in range(m):
                eta = torch.randn(n, d, device=device)
                S_A_x = model(a_t * x + np.sqrt(delta_t) * eta)
                loss = torch.sum((np.sqrt(delta_t) * S_A_x + eta) ** 2) / (d * n)
                total_loss += loss.item() / m
        return total_loss

    # ------------------------------------------------------
    # Test loss
    def compute_test_loss(A):
        batch_size = 100
        N = 100 * n
        test_dataset = torch.randn(N, d, device=device)
        eta = torch.randn(N, d, device=device)
        total_loss = 0.0
        num_batches = N // batch_size
        for i in range(num_batches):
            x_batch = test_dataset[i * batch_size : (i + 1) * batch_size]
            eta_batch = eta[i * batch_size : (i + 1) * batch_size]
            S_A_x = model(a_t * x_batch + np.sqrt(delta_t) * eta_batch)
            loss = torch.sum((np.sqrt(delta_t) * S_A_x + eta_batch) ** 2) / (d * N)
            total_loss += loss.item()
        return total_loss

    # ------------------------------------------------------
    # Error on the Score
    def compute_error_score(A):
        batch_size = 100
        N = 10_000
        test_dataset = torch.randn(N, d, device=device)
        total_loss = 0.0
        num_batches = N // batch_size
        for i in range(num_batches):
            x_batch = test_dataset[i * batch_size : (i + 1) * batch_size]
            S_A_x = model(x_batch)
            loss = torch.sum((S_A_x + x_batch) ** 2) / (d * N)
            total_loss += loss.item()
        return total_loss

    # ------------------------------------------------------
    # Logarithmic sampling of epochs
    def sample_logarithmically(N, T):
        log_samples = np.linspace(np.log(1), np.log(T), N)
        samples = np.exp(log_samples)
        return sorted({int(v) for v in samples})

    sample_times = sample_logarithmically(400, args.num_epochs)

    # ======================================================
    #                    Output Setup
    # ======================================================
    run_name = f"Psi_p={Psi_p:.1f}_Psi_n={Psi_n:.1f}_d={d:.0e}_t={t:.0e}_{args.optimizer}"
    save_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving results to: {save_dir}\n")

    # ======================================================
    #                    Optimizer
    # ======================================================
    optimizer = optim.SGD([A], lr=learning_rate) if args.optimizer == "GD" else optim.Adam([A], lr=learning_rate)

    # ======================================================
    #                    Training Loop
    # ======================================================
    Train_loss, Test_loss, Error_score, Training_time = [], [], [], []

    for epoch in range(args.num_epochs):
        eta = torch.randn(n, d, device=device)
        loss = train_objective(A, x, eta)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Logging at sampled times
        if epoch in sample_times:
            with torch.no_grad():
                train = compute_train_loss(A, x)
                test = compute_test_loss(A)
                error = compute_error_score(A)

                Train_loss.append(train)
                Test_loss.append(test)
                Error_score.append(error)
                Training_time.append(epoch * learning_rate)

            # Save progress
            np.save(os.path.join(save_dir, "Train_loss.npy"), np.array(Train_loss))
            np.save(os.path.join(save_dir, "Test_loss.npy"), np.array(Test_loss))
            np.save(os.path.join(save_dir, "Error_score.npy"), np.array(Error_score))
            np.save(os.path.join(save_dir, "Training_time.npy"), np.array(Training_time))

            print(f"[Epoch {epoch:,}] Train={train:.4e} | Test={test:.4e} | Error on the score={error:.4e}")

    print("\n✅ Training complete. Results saved to:", save_dir)


if __name__ == "__main__":
    main()




