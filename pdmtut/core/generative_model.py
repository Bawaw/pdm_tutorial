import torch

class GenerativeModel(torch.nn.Module):
    def fit_model(self, X, path):
        pass

    def encode(self, X):
        pass

    def decode(self, z):
        pass

    def save(self, path):
        pass

    def load(path):
        pass

    def save_exists(path):
        pass

    def log_likelihood(self, X):
        pass

    def sample_posterior(self, n_samples):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-4, weight_decay=1e-4)
        return {
            'optimizer': optimizer,
            'lr_scheduler':
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, min_lr=2.e-8, factor=0.5, verbose=True, patience=100),
            'monitor': 'train_loss'
        }
