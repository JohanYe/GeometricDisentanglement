# Custom functions with gradient support
import torch
from torch.autograd import Function, gradcheck
import numpy as np
import scipy.special


class Hyp1f1(Function):
    """ Confluent hypergeometric function 1F1.
    Note:     a and b should be scalar
              b should be different to zero
              z should be negative and in float64

    Defintion: https://en.wikipedia.org/wiki/Confluent_hypergeometric_function
    Gradient: d/dx(1F1(a, b, x)) = (a 1F1(a + 1, b + 1, x))/b
    """

    @staticmethod
    def forward(ctx, a, b, z):
        # inputs in torch
        ctx.save_for_backward(a,b, z)
        z_np = z.detach().numpy()
        a = a.detach().numpy()
        b = b.detach().numpy()
        hyp1f1 = torch.from_numpy(scipy.special.hyp1f1(a, b, z_np)).to(z)
        return hyp1f1

    @staticmethod
    def backward(ctx, grad_output):
        a, b, z = ctx.saved_tensors
        grad_a = grad_b = grad_z = None
        if ctx.needs_input_grad[2]:
            a, b, z = a.detach().numpy(), b.detach().numpy(), z.detach().numpy()
            grad_hyp1f1 = (a/b)*scipy.special.hyp1f1(a+1, b+1, z)
            grad_hyp1f1 = torch.from_numpy(grad_hyp1f1).to(grad_output)
            grad_input = grad_output*grad_hyp1f1
        return None, None, grad_input



def sanity_check():
    torch.autograd.set_detect_anomaly(True)
    hyp1f1 = Hyp1f1.apply

    mu = torch.randn(10,3, dtype=torch.float64, requires_grad=True)
    sigma = torch.randn(10, dtype=torch.float64, requires_grad=True)
    a, b = torch.tensor(-1/2), torch.tensor(3/2)
    z = -(1/2)*(mu**2).sum(1)/(torch.abs(sigma))
    
    test = gradcheck(hyp1f1, (a,b,z))
    print('gradcheck:', test)

    def model(mu, sigma):
        z = -(1/2)*(mu**2).sum(1)/torch.abs(sigma)
        output = hyp1f1(a,b,z)
        loss = ((output)**2).sum()
        return loss

    parameters = {'params':mu, 'params':sigma}
    optimizer = torch.optim.Adam([parameters], lr=1)

    def closure():
        optimizer.zero_grad()
        loss = model(mu, sigma)
        loss.backward()
        return loss

    for _ in range(50):
        optimizer.step(closure=closure)
        if torch.max(torch.abs(parameters['params'][0].grad)) < 1e-4: break

    loss = model(mu, sigma)
    # print('loss: {}, mu: {}, sigma: {}'.format(loss, mu, sigma))


if __name__ == '__main__':
    sanity_check()
