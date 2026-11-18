import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedCoordinateAttention(nn.Module):
    """Grouped Coordinate Attention Module as described in the paper."""
    def __init__(self, in_channels, reduction=32, groups=8):
        """
        Args:
            in_channels (int): Number of input channels.
            reduction (int): Reduction ratio 'r' for the bottleneck. Default: 32.
            groups (int): Number of groups 'G'. Default: 8.
        """
        super(GroupedCoordinateAttention, self).__init__()
        self.groups = groups
        self.group_channels = in_channels // groups
        self.reduction = reduction

        # Bottleneck network: shared between H and W directions
        # Input channels for bottleneck is 2 * group_channels (due to max+avg pool)
        bottleneck_channels = max(self.group_channels // reduction, 1) # Ensure at least 1 channel

        # Shared layers for both H and W attention maps
        self.conv_fusion_h = nn.Sequential(
            nn.Conv2d(2 * self.group_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, self.group_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.conv_fusion_w = nn.Sequential(
            nn.Conv2d(2 * self.group_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(bottleneck_channels, self.group_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Output tensor of shape (B, C, H, W).
        """
        b, c, h, w = x.size()
        # Reshape to group channels: (B, G, Cg, H, W)
        x_grouped = x.view(b, self.groups, self.group_channels, h, w)

        # Process each group
        out_groups = []
        for g in range(self.groups):
            xg = x_grouped[:, g, :, :, :] # Shape: (B, Cg, H, W)

            # --- Direction-aware pooling ---
            # Pool along W dimension (for H attention)
            avg_pool_h = F.avg_pool2d(xg, kernel_size=(1, w), stride=1) # Shape: (B, Cg, H, 1)
            max_pool_h = F.max_pool2d(xg, kernel_size=(1, w), stride=1) # Shape: (B, Cg, H, 1)
            # Concatenate avg and max along channel dim -> (B, 2*Cg, H, 1)
            concat_h = torch.cat([avg_pool_h, max_pool_h], dim=1)

            # Pool along H dimension (for W attention)
            avg_pool_w = F.avg_pool2d(xg, kernel_size=(h, 1), stride=1) # Shape: (B, Cg, 1, W)
            max_pool_w = F.max_pool2d(xg, kernel_size=(h, 1), stride=1) # Shape: (B, Cg, 1, W)
            # Concatenate avg and max along channel dim -> (B, 2*Cg, 1, W)
            concat_w = torch.cat([avg_pool_w, max_pool_w], dim=1)

            # --- Shared bottleneck network ---
            # Apply shared conv layers to generate attention maps
            # Attention along H dimension (spatial: H, 1)
            att_h = self.sigmoid(self.conv_fusion_h(concat_h)) # Shape: (B, Cg, H, 1)
            # Attention along W dimension (spatial: 1, W)
            att_w = self.sigmoid(self.conv_fusion_w(concat_w)) # Shape: (B, Cg, 1, W)

            # --- Feature modulation ---
            # Apply attention maps via broadcasting
            # xg: (B, Cg, H, W), att_h: (B, Cg, H, 1), att_w: (B, Cg, 1, W)
            yg = xg * att_h * att_w # Element-wise multiplication

            out_groups.append(yg)

        # Concatenate output groups back along channel dimension
        output = torch.cat(out_groups, dim=1) # Shape: (B, G*Cg, H, W) = (B, C, H, W)

        return output

# Example usage:
if __name__ == "__main__":
    # Example input tensor (Batch=2, Channels=64, Height=224, Width=224)
    input_tensor = torch.randn(2, 8, 224, 224)

    # Create GCA module
    gca_module = GroupedCoordinateAttention(in_channels=8, reduction=32, groups=8)

    # Forward pass
    output_tensor = gca_module(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")
    # Expected output shape: torch.Size([2, 64, 224, 224])

    # Verify the shapes are correct
    assert input_tensor.shape == output_tensor.shape, "Input and output shapes do not match!"
    print("GCA Module forward pass successful!")