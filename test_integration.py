"""Quick ecosystem integration test."""


def test_ecosystem():
    """Quick test to verify all packages work together."""
    print("Testing PyTorch DLN Ecosystem Integration...")

    # Test torch-random
    try:
        import torch.nn as nn
        from torch_random import gibbs

        # Create a simple model and loss for data generation
        simple_model = nn.Sequential(nn.Linear(2, 1))
        loss_fn = nn.MSELoss()

        x, y = gibbs(
            model=simple_model,
            loss_fn=loss_fn,
            n_samples=10,
            input_dim=2,
            noise_std=0.1,
        )

        print("✓ torch-random: Data generation works")
    except Exception as e:
        print(f"✗ torch-random failed: {e}")
        return False

    # Test torch_dln
    try:
        from torch_dln import DLN, zoo

        model1 = DLN([2, 5, 1])
        model2 = zoo.TinyDLN(input_dim=2, output_dim=1)

        # Test forward pass
        model1(x)
        model2(x)

        print("✓ torch_dln: Model construction and zoo works")
    except Exception as e:
        print(f"✗ torch_dln failed: {e}")
        return False

    # Test torch_bdn
    try:
        from torch_bdn import BayesianNet

        bayes_net = BayesianNet(model1)

        # Quick sampling test (just 5 samples)
        from torch_bdn.sampling import SGLD, Sampler

        sampler = Sampler(bayes_net, x, y)
        samples = sampler.sample(SGLD(lr=0.01, n_warmup=5), n_samples=5)

        print("✓ torch_bdn: Bayesian inference works")
        print(f"  Generated {len(samples['parameters'])} posterior samples")
    except Exception as e:
        print(f"✗ torch_bdn failed: {e}")
        return False

    print("\n🎉 Complete ecosystem integration successful!")
    print("All packages working together correctly.")
    return True


if __name__ == "__main__":
    success = test_ecosystem()
    exit(0 if success else 1)
