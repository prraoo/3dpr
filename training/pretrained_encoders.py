import torch
import torch.nn as nn
import timm


class CustomResNet_v0(nn.Module):
    def __init__(self):
        super(CustomResNet_v0, self).__init__()
        # Load a pre-trained ResNet model
        # self.model = timm.create_model('resnet50', pretrained=True)
        self.model = timm.create_model('resnet50.fb_swsl_ig1b_ft_in1k', pretrained=True)

        # Modify the first conv layer to accept 99 input channels
        original_conv1 = self.model.conv1
        self.model.conv1 = nn.Conv2d(
            in_channels=99,
            out_channels=64,
            kernel_size=7,
            stride=1,  # Set stride to 1 to preserve spatial dimensions
            padding=3,
            bias=False
        )
        # Initialize the new conv1 weights
        nn.init.kaiming_normal_(self.model.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Remove the max pooling layer to maintain spatial dimensions
        self.model.maxpool = nn.Identity()

        # Set the stride of convolutional layers to 1 to prevent downsampling
        for layer in [self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]:
            for block in layer:
                # Corrected: Modify conv2.stride instead of conv1.stride
                if hasattr(block, 'conv2'):
                    block.conv2.stride = (1, 1)
                # Adjust the downsample layer if present
                if hasattr(block, 'downsample') and block.downsample is not None:
                    # Ensure the downsample convolution's stride is set to (1, 1)
                    if isinstance(block.downsample, nn.Sequential):
                        if hasattr(block.downsample[0], 'stride'):
                            block.downsample[0].stride = (1, 1)
                    elif isinstance(block.downsample, nn.Conv2d):
                        block.downsample.stride = (1, 1)

        # Replace the fully connected layer with a convolutional layer
        self.model.fc = nn.Identity()
        self.conv_last = nn.Conv2d(
            in_channels=1024,
            out_channels=96,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def forward(self, x):
        # Forward pass through the modified ResNet
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.act1(x)
        # Max pooling is replaced with Identity, so we skip it
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        # x = self.model.layer4(x)
        # Pass through the final convolutional layer
        x = self.conv_last(x)
        return x

# Example usage
if __name__ == "__main__":
    model = CustomResNet_v0()
    input_tensor = torch.randn(4, 99, 256, 256)
    output = model(input_tensor)
    print(f"Final output shape: {output.shape}")  # Should be [4, 96, 256, 256]




