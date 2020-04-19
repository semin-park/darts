import torch
import torch.nn as nn
import torch.nn.functional as F
from .operations import OPS, ReLUConvBN, Identity, ResNetBasicBlock
from .genotypes import BENCH201PRIMITIVES
from torch.autograd import Variable


class MixedOp(nn.Module):

    def __init__(self, C, stride):
        super(MixedOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in BENCH201PRIMITIVES:
            op = OPS[primitive](C, stride, affine=False)
            self._ops.append(op)

    def forward(self, x, weights):
        return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell(nn.Module):

    def __init__(self, steps, C_prev, C):
        super(Cell, self).__init__()
        self._steps = steps
        if C_prev == C:
            self.preprocess = Identity()
            self.reduction = False

            self._ops = nn.ModuleList()
            for i in range(self._steps):
                for j in range(i):
                    op = MixedOp(C, stride=1)
                    self._ops.append(op)
        else:
            self.preprocess = ResNetBasicBlock(C_prev, C, stride=2)
            self.reduction = True

    def forward(self, state, weights):
        state = self.preprocess(state)
        if self.reduction:
            return state
        states = [state]
        offset = 0
        for i in range(self._steps - 1):
            state = sum(self._ops[offset+j](s, weights[offset+j]) for j, s in enumerate(states))
            offset += len(states)
            states.append(state)
        return state


class SequentialNetwork(nn.Module):

    def __init__(self, C, num_classes, layers, criterion, steps=4, stem_multiplier=1, device=None):
        super(SequentialNetwork, self).__init__()
        self._C = C
        self._num_classes = num_classes
        self._layers = layers
        self._criterion = criterion
        self._steps = steps
        self.device = device
        if device is None:
            self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)  # TODO: momentum=0.9 ?
        )
 
        C_prev, C_curr = C_curr, C
        self.cells = nn.ModuleList()
        for i in range(layers):
            if i in [5, 11]:
                C_curr *= 2
            cell = Cell(steps, C_prev, C_curr)
            self.cells.append(cell)
            C_prev = C_curr

        self.last_layer = nn.Sequential(
            nn.BatchNorm2d(C_prev),
            nn.ReLU()
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

        self._initialize_alphas()

    def new(self):
        model_new = SequentialNetwork(self._C, self._num_classes, self._layers, self._criterion).to(self.device)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def forward(self, input):
        state = self.stem(input)
        for i, cell in enumerate(self.cells):
            weights = F.softmax(self.alphas, dim=-1)
            state = cell(state, weights)
        out = self.global_pooling(state)
        logits = self.classifier(out.view(out.size(0),-1))
        return logits

    def _loss(self, input, target):
        logits = self(input)
        return self._criterion(logits, target) 

    def _initialize_alphas(self):
        k = sum(1 for i in range(self._steps) for n in range(i))
        num_ops = len(BENCH201PRIMITIVES)

        self.alphas = Variable(1e-3 * torch.randn(k, num_ops).to(self.device), requires_grad=True)
        self._arch_parameters = [self.alphas]

    def arch_parameters(self):
        return self._arch_parameters
