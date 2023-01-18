# %%
import torch.nn.functional as F
import torch.nn as nn

class OneLayerDecoder(nn.Module):
    # The Conv1DDecoder class is a simple one-layer convolutional neural network that takes an input tensor of shape (batch_size, input_channels, input_dim) and applies a 1D convolution to produce an output tensor of shape (batch_size, output_channels, output_dim).
    # The 1D convolution is defined in the __init__ method of the Conv1DDecoder class using the nn.Conv1d module from PyTorch. This module takes three arguments:
    # input_channels: the number of input channels in the input tensor. In this case, the input tensor has shape (batch_size, 512, 2047), so input_channels is set to 512.
    # output_channels: the number of output channels in the output tensor. In this case, the output tensor should have shape (batch_size, 1, 10240), so output_channels is set to 1.
    # kernel_size: the size of the convolutional kernel. In this case, the kernel size is set to 3, which means that the convolutional operation will consider a window of 3 adjacent values in the input tensor at each position.
    # The forward pass of the Conv1DDecoder class is defined in the forward method. This method takes an input tensor x and applies the 1D convolution defined in the __init__ method to produce an output tensor. The output tensor is then returned.
    def __init__(self, input_channels=512, output_channels=1, output_dim=10240):
        super().__init__()

        # These lines of code define the layers and initialization of the Conv1DDecoder model.
        # The ConvTranspose1d layer is a type of 1D convolutional layer that is used for upsampling the input data. It works by inserting zeros between the elements of the input tensor and then performing a normal 1D convolution operation. The kernel_size, padding, and stride parameters control the shape of the kernel and the spacing between the elements of the input tensor, just like in a regular 1D convolutional layer. The output_padding parameter controls the number of zeros to insert between the elements of the output tensor.
        # The weight and bias parameters of the ConvTranspose1d layer are initialized using the normal_ and zero_ methods of the Tensor class, respectively. The normal_ method initializes the weights with random values drawn from a normal distribution with mean 0 and standard deviation 0.02, while the zero_ method initializes the biases with 0.
        # Finally, the output_dim attribute of the Conv1DDecoder class is initialized with the output_dim parameter, which is the expected size of the output tensor.
        self.deconv1d = nn.ConvTranspose1d(
            input_channels, output_channels,
            kernel_size=3, padding=1, output_padding=1, stride=2)
        self.deconv1d.weight.data.normal_(0, 0.02)
        self.deconv1d.bias.data.zero_()
        self.output_dim = output_dim

    def forward(self, x):
        x = self.deconv1d(x)
        # The F.interpolate function will allow you to resize the output tensor to the desired size of output_dim.
        x = F.interpolate(x, size=self.output_dim,
                          mode='linear', align_corners=False)
        return x



class TwoLayerDecoder(nn.Module):
    def __init__(self, hidden_channels=512, output_channels=1, output_dim=10240):
        super().__init__()
        # These lines of code define the layers and initialization of the Conv1DDecoder model.
        # The ConvTranspose1d layer is a type of 1D convolutional layer that is used for upsampling the input data. 
        # It works by inserting zeros between the elements of the input tensor and then performing a normal 1D convolution operation. 
        # The kernel_size, padding, and stride parameters control the shape of the kernel and the spacing between the elements of the 
        # input tensor, just like in a regular 1D convolutional layer. The output_padding parameter controls the number of zeros to 
        # insert between the elements of the output tensor.

        # The weight and bias parameters of the ConvTranspose1d layer are initialized using the normal_ and zero_ methods of the Tensor class, respectively. The normal_ method initializes the weights with random values drawn from a normal distribution with mean 0 and standard deviation 0.02, while the zero_ method initializes the biases with 0.
        # Finally, the output_dim attribute of the Conv1DDecoder class is initialized with the output_dim parameter, which is the expected size of the output tensor.
        # self.deconv1d.weight.data.normal_(0, 0.02)
        # self.deconv1d.bias.data.zero_()

        #l3^-1 in decoder: 
        # in: [2, 512, 256] 
        # out: [2, 512, 513] WRONG
        self.conv_trans_layer1 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=4, padding=2, output_padding=1, stride=2) 
        
        # out: [2, 512, ?] WRONG
        self.conv_trans_layer2 = nn.ConvTranspose1d(hidden_channels, hidden_channels, kernel_size=8, padding=2, output_padding=2, stride=4)
        
        # [2, 1, 10233]
        self.conv_trans_layer3 = nn.ConvTranspose1d(hidden_channels, output_channels, kernel_size=10, padding=2, output_padding=2, stride=5)

        
        self.output_dim = output_dim

    def forward(self, x):
        x = self.conv_trans_layer1(x)
        # x = nn.ReLU(True),
        x = F.relu(x)
        x = self.conv_trans_layer2(x)

        x = F.relu(x)
        x = self.conv_trans_layer3(x)

        # The F.interpolate function will allow you to resize the output tensor to the desired size of output_dim.
        x = F.interpolate(x, size=self.output_dim, mode='linear', align_corners=False)
        return x
