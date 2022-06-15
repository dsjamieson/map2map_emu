import torch


def adv_model_wrapper(module):
    """Wrap an adversary model to also take lists of Tensors as input,
    to be concatenated along the batch dimension
    """
    class new_module(module):
        def forward(self, x):
            if not isinstance(x, torch.Tensor):
                x = torch.cat(x, dim=0)

            return super().forward(x)

    return new_module


def adv_criterion_wrapper(module):
    """Wrap an adversarial criterion to:
    * also take lists of Tensors as target, used to split the input Tensor
      along the batch dimension
    * expand target shape as that of input
    * return a list of losses, one for each pair of input and target Tensors
    """
    class new_module(module):
        def forward(self, input, target):
            assert isinstance(input, torch.Tensor)

            if isinstance(target, torch.Tensor):
                input = [input]
                target = [target]
            else:
                input = self.split_input(input, target)
            assert len(input) == len(target)

            target = [t.expand_as(i) for i, t in zip(input, target)]

            loss = [super(new_module, self).forward(i, t)
                    for i, t in zip(input, target)]

            return loss

        @staticmethod
        def split_input(input, target):
            assert all(t.dim() == target[0].dim() > 0 for t in target)
            if all(t.shape[0] == 1 for t in target):
                size = input.shape[0] // len(target)
            else:
                size = [t.shape[0] for t in target]

            return torch.split(input, size, dim=0)

    return new_module
