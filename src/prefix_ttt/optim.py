"""AdamW whose FP32 master lives in optimizer state, where ZeRO can shard it.

A BF16 parameter cannot accumulate the updates this recipe needs: at ``|w|~0.02``
one BF16 ulp is about ``1.2e-4`` while a step moves the weight by roughly the
learning rate, ``2e-5`` -- most steps would round to nothing. The usual fix is an
FP32 master copy, but ZeRO only shards what lives in *optimizer state*: a master
kept as a module parameter would be replicated to every rank by the post-step
broadcast that ZeRO performs on its parameters.

Keeping the master in ``state`` is therefore what makes it shardable, and it keeps
the module's own storage BF16 -- the dtype the forward already computes in. The
update arithmetic mirrors ``torch.optim.AdamW`` exactly, so the two can be
compared bit for bit on FP32 parameters.
"""
import torch


class MasterWeightAdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps, weight_decay = group['eps'], group['weight_decay']
            for parameter in group['params']:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state['step'] = 0
                    state['master'] = parameter.detach().float().clone()
                    state['exp_avg'] = torch.zeros_like(state['master'])
                    state['exp_avg_sq'] = torch.zeros_like(state['master'])
                master, exp_avg, exp_avg_sq = state['master'], state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                grad = parameter.grad.detach().float()
                exp_avg.lerp_(grad, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                denom = (exp_avg_sq.sqrt() / bias_correction2 ** 0.5).add_(eps)
                if weight_decay:
                    master.mul_(1 - lr * weight_decay)
                master.addcdiv_(exp_avg, denom, value=-lr / bias_correction1)
                parameter.copy_(master)
        return loss
