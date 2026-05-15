import __main__

import torch
import torch.nn as nn
import numpy as np

class SineActivation(nn.Module):
    """Custom sine activation."""

    def forward(self, x):
        return torch.sin(x)


class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0    = omega_0
        self.in_features = in_features
        self.linear     = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights(is_first)

    def init_weights(self, is_first):
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(
                    -1 / self.in_features,
                     1 / self.in_features
                )
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                     np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))


class ResidualSineLayer(nn.Module):
    """
    Residual refinement block:
    Input -> Linear -> Sine -> Linear -> Sine
           -> Residual Add
    """

    def __init__(self, dim=120):
        super().__init__()

        self.linear1 = nn.Linear(dim, dim)
        self.act1 = SineActivation()

        self.linear2 = nn.Linear(dim, dim)
        self.act2 = SineActivation()
    
    def init_weights(self):
        with torch.no_grad():
            self.linear1.weight.uniform_(
                -np.sqrt(6 / self.in_features)   / self.omega_0,
                 np.sqrt(6 / self.in_features)   / self.omega_0
            )
            self.linear2.weight.uniform_(
                -np.sqrt(6 / self.in_features) / self.omega_0,
                 np.sqrt(6 / self.in_features) / self.omega_0
            )
           

    def forward(self, x):
        identity = x

        out = self.linear1(x)
        out = self.act1(out)

        out = self.linear2(out)
        out = self.act2(out)

        out = out + identity

        return out


class MyResidualSineNet(nn.Module):
    """
    Architecture:

    Step 1: Input (3D coordinates)
    Step 2: SineLayer (3 -> 120)
    Step 3: ResidualSineLayer x10 (120 -> 120)
    Step 4: Final Linear (120 -> 1)
    Step 5: Scalar Output
    """

    def __init__(self, input_dim=3, hidden_dim=120, num_residual_blocks=10, output_dim=1, omega_0=30):
        super().__init__()

        # Initial sine feature extractor
        self.input_layer = SineLayer(input_dim, hidden_dim, omega_0=omega_0)

        # Residual refinement blocks
        self.residual_blocks = nn.Sequential(*[ResidualSineLayer(hidden_dim) for _ in range(num_residual_blocks)])

        # Final prediction head
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.residual_blocks(x)
        x = self.output_layer(x)

        return x


if __name__ == __main__:
    # Example usage
    model = MyResidualSineNet()
    print(model)



