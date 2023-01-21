import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden, kernel_sizes, strides, padding):
        super(Encoder, self).__init__()

        self.hidden = hidden

        assert (
            len(kernel_sizes) == len(strides) == len(padding)
        ), "Inconsistent size of network parameters (kernels, strides and padding)"

        self.model = nn.Sequential()

        for idx in range(len(kernel_sizes)):
            self.model.add_module(
                "layer {}".format(idx),
                self.new_block(
                    input_dim,
                    self.hidden,
                    kernel_sizes[idx],
                    strides[idx],
                    padding[idx],
                ),
            )
            input_dim = self.hidden

    def new_block(self, in_dim, out_dim, kernel_size, stride, padding):
        new_block = nn.Sequential(
            nn.Conv1d(
                in_dim, out_dim, kernel_size=kernel_size, stride=stride, padding=padding
            ),
            nn.ReLU(),
        )
        return new_block

    def forward(self, x):
        # print("****")
        # self.model.eval()
      
        # print(self.model)

        # for i, module in enumerate(self.model.modules()):
        #     if isinstance(module, (nn.Conv1d, nn.Linear)):
        #         print("Layer ", i)
        #         print("Weights: ", module.weight)
        #         print("Bias: ", module.bias)
        #     else:
        #         print("no")
                
        # # for layer in self.model:
        # #     w = layer.weight.item()
        # #     bias = layer.bias.item()
    
        # # for i, layer in enumerate(self.model):
        # #     print("Layer ", i)
        # #     print("Weights: ", layer.weight)
        # #     print("Bias: ", layer.bias)
            
    


        return self.model(x)

    def forward_through_n_layers(self, x, n):
        for i in range(n):
            x = self.model[i](x)
        return x
