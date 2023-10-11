import torch
import time
import evaluate
from .logging_utils import Averager
from datasets.iterable_dataset import IterableDataset
from tqdm import tqdm
import os
import selfies as sf
import re

def filter_selfies(s):
    pattern = r'(\[[^\]]+\]\.?)'
    matches = re.findall(pattern, s)
    return ''.join(matches)

def maybe_save_checkpoint(accelerator, args):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.checkpoint.every_steps == 0
    ):
        output_dir = f'checkpoint-{args.mode}-{args.current_train_step}'
        accelerator.save_state(output_dir=output_dir)


def maybe_eval_predict(model, dataloader, logger, args, tokenizer, accelerator, prefix='test'):
    if (
        args.current_train_step > args.optim.total_steps
        or args.current_train_step % args.pred.every_steps == 0
    ):
        model.eval()

        with torch.no_grad():
            # eval(model, dataloader, logger, args, tokenizer, accelerator)

            if args.mode == 'ft':
                predict(
                    model, dataloader, logger, args, tokenizer, accelerator, prefix=prefix
                )

        args.last_log = time.time()
        model.train()


def maybe_logging(averager, args, model, optimizer, logger):
    if args.current_train_step % args.logging.every_steps == 0:
        stats = extra_stats(args, model, optimizer)

        seconds_per_step = (time.time() - args.last_log) / args.logging.every_steps
        stats['seconds_per_step'] = seconds_per_step

        averager.update(stats)
        averaged_stats = averager.average()

        logger.log_stats(
            stats=averaged_stats,
            step=args.current_train_step,
            args=args,
            prefix='train/'
        )

        args.last_log = time.time()


def maybe_grad_clip_and_grad_calc(accelerator, model, args):
    if args.logging.grad_l2:
        grad_l2 = (
            sum(p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        )
    else:
        grad_l2 = None

    if args.optim.grad_clip > 0:
        accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=args.optim.grad_clip,
            norm_type=2,
        )

    if grad_l2 is not None:
        return {'grad_l2': grad_l2}
    else:
        return {}


def extra_stats(args, model, optimizer):
    stats = {}

    if args.logging.weights_l2:
        weights_l2 = sum(p.detach().norm(2).item() ** 2 for p in model.parameters()) ** 0.5
        stats['weights_l2'] = weights_l2

    cur_lr = optimizer.param_groups[0]['lr']
    stats['lr'] = cur_lr

    return stats


def forward(model, batch, calc_acc=False):
    outputs = model(**batch)
    loss = outputs.loss

    stats = {}
    stats['loss'] = loss.detach().float().item()

    if calc_acc:
        correct = (outputs.logits.argmax(-1) == batch["labels"]).sum().item()
        accuracy = correct / batch["labels"].numel()
        stats['accuracy'] = accuracy

    return loss, stats


def eval(model, dataloader, logger, args, tokenizer, accelerator):
    args.last_log = time.time()
    averager = Averager()

    for batch_id, batch in enumerate(dataloader, start=1):
        batch = batch.to(accelerator.device)
        if batch_id == args.eval.corrected_steps * args.optim.grad_acc:
            break

        _, stats = forward(model, batch, calc_acc=True)
        averager.update(stats)

    averager.update({'time': time.time() - args.last_log})
    averaged_stats = averager.average()

    logger.log_stats(
        stats=averaged_stats,
        step=args.current_train_step,
        args=args,
        prefix='eval/'
    )


def predict(model, dataloader, logger, args, tokenizer, accelerator, prefix='test'):
    args.last_log = time.time()

    if args.test_task == 'mol2text':
        metric = evaluate.load(os.path.join(__file__.split('BioT5/utils')[0], 'BioT5/metrics/translation_metrics'))
    elif args.test_task == 'text2mol':
        metric = evaluate.load(os.path.join(__file__.split('BioT5/utils')[0], 'BioT5/metrics/save_only_metrics'))
    elif args.test_task == 'dti' or args.test_task == 'molnet':
        metric = evaluate.load(os.path.join(__file__.split('BioT5/utils')[0], 'BioT5/metrics/dti_metrics'))
    elif args.test_task == 'ppi':
        metric = evaluate.load(os.path.join(__file__.split('BioT5/utils')[0], 'BioT5/metrics/ppi_metrics'))
    else:
        raise NotImplementedError
    
    samples_seen = 0
    selfies_invalid = 0

    def decode(preds):
        preds[preds == -100] = tokenizer.pad_token_id
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        preds = [pred.strip() for pred in preds]
        return preds

    for step, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        batch = batch.to(accelerator.device)
        if args.test_task == 'dti' or args.test_task == 'molnet' or args.test_task == 'ppi':
            generation_results = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=args.data.max_target_len,
                generation_config=model.generation_config,
                return_dict_in_generate=True,
                output_scores=True,
            )
            predictions, scores = generation_results.sequences, generation_results.scores
        else:
            predictions = model.generate(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                max_length=args.data.max_target_len,
                generation_config=model.generation_config,
            )
        predictions = decode(predictions)
        references = decode(batch["labels"])
        inputs = decode(batch["input_ids"])
        if args.test_task == 'mol2text':
            inputs = [sf.decoder(input_i.split('- Input: ')[-1].split(' Output:')[0]) for input_i in inputs]
            references = [(references[i], inputs[i]) for i in range(len(references))]
        elif args.test_task == 'text2mol':
            inputs = [input_i.split('- Input: ')[-1].split(' Output:')[0] for input_i in inputs]
            for i in range(len(predictions)):
                try: 
                    predictions[i] = sf.decoder(predictions[i])
                except:
                    predictions[i] = sf.decoder(filter_selfies(predictions[i]))
                    selfies_invalid += 1
            references = [sf.decoder(ref_i) for ref_i in references]
            references = [(references[i], inputs[i]) for i in range(len(references))]
        elif args.test_task == 'dti' or args.test_task == 'molnet' or args.test_task == 'ppi':
            # No: 465, Yes: 2163
            predictions = [(scores[0][i][2163] / (scores[0][i][2163] + scores[0][i][465])).item() for i in range(len(predictions))]
        else:
            raise NotImplementedError

        # If we are in a multiprocess environment, the last batch has duplicates
        if step == len(dataloader) - 1:
            predictions = predictions[: len(dataloader.dataset) - samples_seen]
            references = references[: len(dataloader.dataset) - samples_seen]
        else:
            samples_seen += len(references)

        metric.add_batch(
            predictions=predictions,
            references=references,
        )

        # TODO for debug
        # if step == 20:
        #     break

    eval_metric = metric.compute(tsv_path=os.path.join(args.working_dir, args.result_fn))

    if args.test_task == 'mol2text':
        logger.log_stats(
            stats={
                "bleu2": eval_metric["bleu2"],
                "bleu4": eval_metric["bleu4"],
                "rouge1": eval_metric["rouge1"],
                "rouge2": eval_metric["rouge2"],
                "rougeL": eval_metric["rougeL"],
                "meteor": eval_metric["meteor"],
                "time": time.time() - args.last_log,
            },
            step=args.current_train_step,
            args=args,
            prefix=f"{prefix}/",
        )
    elif args.test_task == 'text2mol':
        logger.log_stats(
            stats={
                "bleu": eval_metric["bleu"],
                "exact_match": eval_metric["exact_match"],
                "levenshtein": eval_metric["levenshtein"],
                "validity": eval_metric["validity"],
                "invalid selfies num": selfies_invalid,
                "time": time.time() - args.last_log,
            },
            step=args.current_train_step,
            args=args,
            prefix=f"{prefix}/",
        )
    elif args.test_task == 'dti' or args.test_task == 'molnet':
        logger.log_stats(
            stats={
                "accuracy": eval_metric["accuracy"],
                "auroc": eval_metric["auroc"],
                "auprc": eval_metric["auprc"],
                "sensitivity": eval_metric["sensitivity"],
                "specificity": eval_metric["specificity"],
                "f1": eval_metric["f1"],
                "thred_optim": eval_metric["thred_optim"],
                "precision": eval_metric["precision"],
                "time": time.time() - args.last_log,
            },
            step=args.current_train_step,
            args=args,
            prefix=f"{prefix}/",
        )
    elif args.test_task == 'ppi':
        logger.log_stats(
            stats={
                "accuracy": eval_metric["accuracy"],
                "specificity": eval_metric["specificity"],
                "sensitivity": eval_metric["sensitivity"],
                "precision": eval_metric["precision"],
                "mcc": eval_metric["mcc"],
                "auroc": eval_metric["auroc"],
                "f1": eval_metric["f1"],
                "auprc": eval_metric["auprc"],
                "kappa": eval_metric["kappa"],
                "time": time.time() - args.last_log,
            },
            step=args.current_train_step,
            args=args,
            prefix=f"{prefix}/",
        )
    else:
        raise NotImplementedError


def train(model, train_dataloader, validation_dataloader, test_dataloader, accelerator, lr_scheduler,
          optimizer, logger, args, tokenizer):
    model.train()

    train_averager = Averager()

    while args.current_train_step <= args.optim.total_steps:
        if isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(args.current_epoch)

        # In case there is a remainder from previous epoch, we need to reset the optimizer
        optimizer.zero_grad(set_to_none=True)

        for batch_id, batch in enumerate(train_dataloader, start=1):
            if args.current_train_step > args.optim.total_steps:
                break

            loss, stats = forward(model, batch)
            accelerator.backward(loss / args.optim.grad_acc)
            train_averager.update(stats)

            if batch_id % args.optim.grad_acc == 0:
                stats = maybe_grad_clip_and_grad_calc(accelerator, model, args)
                train_averager.update(stats)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                maybe_logging(train_averager, args, model, optimizer, logger)
                
                if accelerator.is_main_process:
                    maybe_eval_predict(accelerator.unwrap_model(model), validation_dataloader, logger, args, tokenizer, accelerator, prefix='validation')
                    maybe_eval_predict(accelerator.unwrap_model(model), test_dataloader, logger, args, tokenizer, accelerator, prefix='test')
                    maybe_save_checkpoint(accelerator, args)
                accelerator.wait_for_everyone()
                args.current_train_step += 1

        args.current_epoch += 1

    
    if accelerator.is_main_process:
        maybe_eval_predict(accelerator.unwrap_model(model), validation_dataloader, logger, args, tokenizer, accelerator, prefix='validation')
        maybe_eval_predict(accelerator.unwrap_model(model), test_dataloader, logger, args, tokenizer, accelerator, prefix='test')
        maybe_save_checkpoint(accelerator, args)
    accelerator.wait_for_everyone()
