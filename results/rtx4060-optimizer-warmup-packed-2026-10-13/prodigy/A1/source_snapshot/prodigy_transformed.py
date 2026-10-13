def step(self, closure=None):
    _packed_old_d = self.param_groups[0]['d']
    'Performs a single optimization step.\n\n    Arguments:\n        closure (callable, optional): A closure that reevaluates the model\n            and returns the loss.\n    '
    loss = None
    if closure is not None:
        loss = closure()
    d_denom = 0.0
    group = self.param_groups[0]
    use_bias_correction = group['use_bias_correction']
    beta1, beta2 = group['betas']
    beta3 = group['beta3']
    if beta3 is None:
        beta3 = math.sqrt(beta2)
    k = group['k']
    d = group['d']
    d_max = group['d_max']
    d_coef = group['d_coef']
    lr = max((group['lr'] for group in self.param_groups))
    if use_bias_correction:
        bias_correction = (1 - beta2 ** (k + 1)) ** 0.5 / (1 - beta1 ** (k + 1))
    else:
        bias_correction = 1
    dlr = d * lr * bias_correction
    growth_rate = group['growth_rate']
    decouple = group['decouple']
    fsdp_in_use = group['fsdp_in_use']
    d_numerator = group['d_numerator']
    d_numerator *= beta3
    for group in self.param_groups:
        decay = group['weight_decay']
        k = group['k']
        eps = group['eps']
        group_lr = group['lr']
        d0 = group['d0']
        safeguard_warmup = group['safeguard_warmup']
        if group_lr not in [lr, 0.0]:
            raise RuntimeError(f'Setting different lr values in different parameter groups is only supported for values of 0')
        for p in group['params']:
            with _packed_controller.parameter(p, 'init'):
                if p.grad is None:
                    continue
                if hasattr(p, '_fsdp_flattened'):
                    fsdp_in_use = True
                grad = p.grad.data
                if decay != 0 and (not decouple):
                    grad.add_(p.data, alpha=decay)
                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['s'] = torch.zeros_like(p.data).detach()
                    state['p0'] = p.detach().clone()
                    state['exp_avg'] = torch.zeros_like(p.data).detach()
                    state['exp_avg_sq'] = torch.zeros_like(p.data).detach()
                exp_avg, exp_avg_sq = (state['exp_avg'], state['exp_avg_sq'])
                s = state['s']
                p0 = state['p0']
                if group_lr > 0.0:
                    d_numerator += d / d0 * dlr * torch.dot(grad.flatten(), (p0.data - p.data).flatten()).item()
                    if safeguard_warmup:
                        s.mul_(beta3).add_(grad, alpha=d / d0 * d)
                    else:
                        s.mul_(beta3).add_(grad, alpha=d / d0 * dlr)
                    d_denom += s.abs().sum().item()
            exp_avg = exp_avg_sq = denom = None
    d_hat = d
    if d_denom == 0:
        return loss
    if lr > 0.0:
        if fsdp_in_use:
            dist_tensor = torch.zeros(2).cuda()
            dist_tensor[0] = d_numerator
            dist_tensor[1] = d_denom
            dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
            global_d_numerator = dist_tensor[0]
            global_d_denom = dist_tensor[1]
        else:
            global_d_numerator = d_numerator
            global_d_denom = d_denom
        d_hat = d_coef * global_d_numerator / global_d_denom
        if d == group['d0']:
            d = max(d, d_hat)
        d_max = max(d_max, d_hat)
        d = min(d_max, d * growth_rate)
    for group in self.param_groups:
        group['d_numerator'] = global_d_numerator
        group['d_denom'] = global_d_denom
        group['d'] = d
        group['d_max'] = d_max
        group['d_hat'] = d_hat
        decay = group['weight_decay']
        k = group['k']
        eps = group['eps']
        for p in group['params']:
            with _packed_controller.parameter(p, 'step'):
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                exp_avg, exp_avg_sq = (state['exp_avg'], state['exp_avg_sq'])
                state['step'] += 1
                if group['lr'] > 0:
                    exp_avg.mul_(beta1).add_(grad, alpha=_packed_old_d * (1 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=_packed_old_d * _packed_old_d * (1 - beta2))
                denom = exp_avg_sq.sqrt().add_(d * eps)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)
                p.data.addcdiv_(exp_avg, denom, value=-dlr)
            exp_avg = exp_avg_sq = denom = None
        group['k'] = k + 1
    return loss