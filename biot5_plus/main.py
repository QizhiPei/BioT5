from accelerate import Accelerator
from omegaconf import open_dict
import hydra
import torch
import time

from utils import (
    setup_basics,
    train,
    predict,
    eval,
    get_lr_scheduler,
    get_optimizer,
    get_tokenizer,
    get_tokenizer_pred,
    get_model,
    get_dataloaders,
    get_config,
)


@hydra.main(config_path="configs", config_name="default", version_base='1.1')
def main(args):
    if args.logging.wandb:
        accelerator = Accelerator(log_with="wandb", cpu=args.device == "cpu")
        def flatten_dict(d, parent_key='', sep='_'):
            items = {}
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.update(flatten_dict(v, new_key, sep=sep))
                else:
                    items[new_key] = v
            return items
        accelerator.init_trackers(
            args.logging.wandb_project,
            config=flatten_dict(args),
            init_kwargs={
                "wandb": {
                    "group": args.logging.wandb_group,
                    "name": args.logging.wandb_run_name,
                    "id": args.logging.wandb_run_name,
                }
            },
        )
    else:
        accelerator = Accelerator(cpu=args.device == "cpu")
    logger = setup_basics(accelerator, args)
    config = get_config(args)
    tokenizer = get_tokenizer(args)
    tokenizer_pred = get_tokenizer_pred(args, tokenizer)
    if args.model.no_plus:
        tokenizer = tokenizer_pred
    model = get_model(args, config, tokenizer, logger)
    optimizer = get_optimizer(model, args)
    # lr_scheduler = get_lr_scheduler(optimizer, args, logger)
    if args.mode == 'pt':
        train_dataloader, validation_dataloader = get_dataloaders(tokenizer, tokenizer_pred, config, args)
    elif args.mode == 'ft':
        train_dataloader, validation_dataloader, test_dataloader = get_dataloaders(tokenizer, tokenizer_pred, config, args)

    logger.log_args(args)

    (
        model,
        optimizer,
        train_dataloader,
        validation_dataloader,
    ) = accelerator.prepare(
        model, optimizer, train_dataloader, validation_dataloader
    )

    if args.mode == 'ft' and args.optim.epochs > 0:
        args.optim.total_steps = len(train_dataloader) * args.optim.epochs // args.optim.grad_acc
        args.optim.warmup_steps = int(0.06 * args.optim.total_steps)
        if args.checkpoint.every_epochs > 0:
            args.checkpoint.every_steps = args.checkpoint.every_epochs * len(train_dataloader) // args.optim.grad_acc
        if args.pred.every_epochs > 0:
            args.pred.every_steps = args.pred.every_epochs * len(train_dataloader) // args.optim.grad_acc

    lr_scheduler = get_lr_scheduler(optimizer, args, logger)

    if args.model.compile:
        model = torch.compile(model)

    with open_dict(args):
        args.current_train_step = 1
        args.current_epoch = 1
        args.last_log = time.time()
    
    if args.restore_train:
        if args.mode == 'pt':
            accelerator.load_state(args.restore_train_path)
            restore_train_step = int(args.restore_train_path.split('-')[-1].strip('/'))
            logger.log_message(f"Restore training from {args.restore_train_path}")
            with open_dict(args):
                args.current_train_step = restore_train_step + 1
            for _ in range(restore_train_step):
                lr_scheduler.step()
        elif args.mode == 'ft':
            ## load the latest checkpoint
            import os
            ckpt_list = os.listdir()
            ckpt_list = [d for d in ckpt_list if d.startswith('checkpoint')]
            if len(ckpt_list) > 0:
                # sort by the number in the prefix
                ckpt_list.sort(key=lambda x: int(x.split('-')[-1]), reverse=True)
                last_ckpt = ckpt_list[0]
                accelerator.load_state(last_ckpt)
                restore_train_step = int(last_ckpt.split('-')[-1])
                logger.log_message(f"Restore training from {last_ckpt}")
                with open_dict(args):
                    args.current_train_step = restore_train_step + 1
                for _ in range(restore_train_step):
                    lr_scheduler.step()

    if args.eval_only:
        model.eval()
        with torch.no_grad():
            eval(model, validation_dataloader, logger, args, tokenizer)
    elif args.predict_only:
        model.eval()
        with torch.no_grad():
            predict(model, test_dataloader, logger,
                    args, tokenizer_pred, accelerator)

    else:
        if args.mode == 'pt':
            train(model, train_dataloader, validation_dataloader, None, accelerator,
                lr_scheduler, optimizer, logger, args, tokenizer, None)
        elif args.mode == 'ft':
            train(model, train_dataloader, validation_dataloader, test_dataloader, accelerator,
                lr_scheduler, optimizer, logger, args, tokenizer, tokenizer_pred)
        else:
            raise NotImplementedError
        
    logger.finish()

if __name__ == "__main__":
    main()
