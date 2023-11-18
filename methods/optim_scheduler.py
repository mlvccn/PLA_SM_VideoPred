
from ast import arg
from timm.scheduler.cosine_lr import CosineLRScheduler
def get_optim_scheduler(args, epoch, model, steps_per_epoch):
    from torch import optim
    lr =args.lr
    optimizer = optim.Adam(filter(lambda x: x.requires_grad is not False ,model.parameters()), lr=lr)
    sched_lower = args.sched.lower()
    total_steps = epoch * steps_per_epoch
    by_epoch = True
    if sched_lower == 'onecycle':
        lr_scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr,
            total_steps=total_steps,
            final_div_factor=getattr(args, 'final_div_factor', 1e4))
        by_epoch = False
    elif sched_lower == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epoch,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=args.warmup_epoch,
            t_in_epochs=True,  # update lr by_epoch
            )
    #optimizer = optim.Adam(model.parameters(), lr=lr)
    
    #scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=epoch * steps_per_epoch)
    #scheduler = CosineLRScheduler(optimizer, t_initial=epoch, warmup_lr_init=1e-5, warmup_t=5, t_in_epochs=True)
    return optimizer, lr_scheduler, by_epoch
