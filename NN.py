import torch as tc
import torch.nn as nn
import copy

class LRP_product(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, f1,f2):
        self.f1, self.f2 = f1, f2
        self.y = f1*f2
        return self.y


    def relprop(self,R):
        #these are not actually relevances -> only at multiplication at the end
        
        R1 = self.f1 * R/ (self.y + 1e-9 * tc.sign(self.y))
        R1 = tc.nan_to_num(R1, 0.0)
        R2 = self.f2

        return R1.detach(), R2.detach() 


class LRP_Linear(nn.Module):
    def __init__(self, inp, outp, gamma=0.01, eps=1e-5):
        super(LRP_Linear, self).__init__()
        self.A_dict = {}
        self.linear = nn.Linear(inp, outp)
        nn.init.kaiming_uniform_(self.linear.weight)
        #nn.init.kaiming_uniform_(self.linear.weight, gain=nn.init.calculate_gain('relu'))
        self.gamma = tc.tensor(gamma)
        self.eps = tc.tensor(eps)
        self.rho = None
        self.iteration = None

    def forward(self, x):

        if not self.training:
            self.A_dict[self.iteration] = x.clone()
        return self.linear(x)

    def relprop(self, R):
        device = next(self.parameters()).device

        A = self.A_dict[self.iteration].clone()
        A, self.eps = A.to(device), self.eps.to(device)

        Ap = A.clamp(min=0).detach().data.requires_grad_(True)
        Am = A.clamp(max=0).detach().data.requires_grad_(True)


        zpp = self.newlayer(1).forward(Ap)  
        zmm = self.newlayer(-1, no_bias=True).forward(Am) 

        zmp = self.newlayer(1, no_bias=True).forward(Am) 
        zpm = self.newlayer(-1).forward(Ap) 

        with tc.no_grad():
            Y = self.forward(A).data

        sp = ((Y > 0).float() * R / (zpp + zmm + self.eps * ((zpp + zmm == 0).float() + tc.sign(zpp + zmm)))).data # new version
        sm = ((Y < 0).float() * R / (zmp + zpm + self.eps * ((zmp + zpm == 0).float() + tc.sign(zmp + zpm)))).data

        (zpp * sp).sum().backward()
        cpp = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zpm * sm).sum().backward()
        cpm = Ap.grad
        Ap.grad = None
        Ap.requires_grad_(True)

        (zmp * sm).sum().backward()
        cmp = Am.grad
        Am.grad = None
        Am.requires_grad_(True)

        (zmm * sp).sum().backward()
        cmm = Am.grad
        Am.grad = None
        Am.requires_grad_(True)


        R_1 = (Ap * cpp).data
        R_2 = (Ap * cpm).data
        R_3 = (Am * cmp).data
        R_4 = (Am * cmm).data


        return R_1 + R_2 + R_3 + R_4

    def newlayer(self, sign, no_bias=False):

        if sign == 1:
            rho = lambda p: p + self.gamma * p.clamp(min=0) # Replace 1e-9 by zero
        else:
            rho = lambda p: p + self.gamma * p.clamp(max=0) # same here

        layer_new = copy.deepcopy(self.linear)

        try:
            layer_new.weight = nn.Parameter(rho(self.linear.weight))
        except AttributeError:
            pass

        try:
            layer_new.bias = nn.Parameter(self.linear.bias * 0 if no_bias else rho(self.linear.bias))
        except AttributeError:
            pass

        return layer_new




class LRP_ReLU(nn.Module):
    def __init__(self):
        super(LRP_ReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x)

    def relprop(self, R):
        return R


class LRP_DropOut(nn.Module):
    def __init__(self, p):
        super(LRP_DropOut, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        return self.dropout(x)

    def relprop(self, R):
        return R


class Model(nn.Module):
    def __init__(self, inp, outp, hidden, hidden_depth, dropout, gamma):
        super(Model, self).__init__()
        self.layers = nn.Sequential(LRP_DropOut(p = 0.0), LRP_Linear(inp, hidden, gamma=gamma), LRP_ReLU())
        for i in range(hidden_depth):
            self.layers.add_module('dropout', LRP_DropOut(p = dropout))
            self.layers.add_module('LRP_Linear' + str(i + 1), LRP_Linear(hidden, hidden, gamma=gamma))
            self.layers.add_module('LRP_ReLU' + str(i + 1), LRP_ReLU())

        self.layers.add_module('dropout', LRP_DropOut(p = dropout))
        self.layers.add_module('LRP_Linear_last', LRP_Linear(hidden, outp, gamma=gamma))
        #self.layers.add_module('dropout', LRP_DropOut(p = dropout)) #new


    def forward(self, x):
        return self.layers.forward(x)

    def relprop(self, R):
        assert not self.training, 'relprop does not work during training time'
        for module in self.layers[::-1]:
            R = module.relprop(R)
        return R

class LinearModel(nn.Module):
    def __init__(self, inp, outp, hidden, hidden_depth, dropout, gamma):
        super(LinearModel, self).__init__()
        self.layers = nn.Sequential(LRP_Linear(inp, outp, gamma=gamma),LRP_DropOut(p = dropout))

    def forward(self, x):
        return self.layers.forward(x)

    def relprop(self, R):
        assert not self.training, 'relprop does not work during training time'
        for module in self.layers[::-1]:
            R = module.relprop(R)
        return R



class Interaction_Model(nn.Module):
    classname = 'Interaction Model'
    def __init__(self, ds):
        super().__init__()
        
        
        self.nfeatures1, self.nfeatures2, self.nfeatures_product, self.nfeatures_out = ds.ndrug_features, ds.nmolecular_features, 1000, 1   #1000

        self.nn1 = Model(self.nfeatures1, self.nfeatures_product, 5000, hidden_depth = 0, dropout = 0.05, gamma = 0.01) #2000
        self.nn2 = Model(self.nfeatures2, self.nfeatures_product, 10000, hidden_depth = 0, dropout = 0.05, gamma = 0.01) #4000

        #self.last_nn = LRP_Linear(self.nfeatures_product,1)

        self.product = LRP_product()
        
        self.dropout = LRP_DropOut(p=0.05)


    def forward(self, drug, molecular):
 
        intermediate1 = self.nn1(drug)
        intermediate2 = self.nn2(molecular)

        product = self.product.forward(intermediate1, intermediate2)

        #outcome = self.last_nn(product)
        outcome = product.mean(axis=1).unsqueeze(1)

        return outcome



    def get_product(self, drug, molecular):
 
        intermediate1 = self.nn1(drug)
        intermediate2 = self.nn2(molecular)

        product = self.product.forward(intermediate1, intermediate2)


        return product

    def relprop(self, R):
        device = R.device
        #product_relevance = self.last_nn.relprop(R)
        
        factor1_relevance, factor2_relevance = self.product.relprop(R)


        input_relevance = tc.zeros(R.shape[0], self.nfeatures1, self.nfeatures2, device=device)

        for i in range(self.nfeatures_product):

            input1_relevance = self.nn1.relprop(factor1_relevance * tc.eye(self.nfeatures_product, device = device)[i])         
            input2_relevance = self.nn2.relprop(factor2_relevance * tc.eye(self.nfeatures_product, device=device)[i])    

            input_relevance += input1_relevance[:,:,None] * input2_relevance[:,None,:]

        return input_relevance

