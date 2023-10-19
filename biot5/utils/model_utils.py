import torch
import datasets
from torch.utils.data import DataLoader, IterableDataset
from omegaconf import open_dict
from datasets.iterable_dataset import IterableDataset
from transformers import (
    AutoTokenizer,
    T5ForConditionalGeneration,
    AutoConfig,
)

from .copied_utils import (
    compute_input_and_target_lengths,
    DataCollatorForT5MLM,
    tokenize_function,
    DataCollatorForNI,
)
from .custom_utils import (
    tokenize_function_seq_desc,
    DataCollatorForUnimptT5,
)
import os
import itertools
class MixedDataset(IterableDataset):
    def __init__(self, dataset_text, dataset_molecule, dataset_protein, dataset_incontext, dataset_mol_text, dataset_pro_text):
        self.dataset_text = dataset_text
        self.dataset_molecule = dataset_molecule
        self.dataset_protein = dataset_protein
        self.dataset_incontext = dataset_incontext
        self.dataset_mol_text = dataset_mol_text
        self.dataset_pro_text = dataset_pro_text

    def __iter__(self):
        text_iter = iter(self.dataset_text)
        molecule_iter = iter(self.dataset_molecule)
        protein_iter = iter(self.dataset_protein)
        incontext_iter = iter(self.dataset_incontext)
        mol_text_iter = iter(self.dataset_mol_text)
        pro_text_iter = iter(self.dataset_pro_text)

        while True:
            try:
                text_batch = next(text_iter)
            except StopIteration:
                text_iter = iter(self.dataset_text)
                text_batch = next(text_iter)

            try:
                molecule_batch = next(molecule_iter)
            except StopIteration:
                molecule_iter = iter(self.dataset_molecule)
                molecule_batch = next(molecule_iter)

            try:
                protein_batch = next(protein_iter)
            except StopIteration:
                protein_iter = iter(self.dataset_protein)
                protein_batch = next(protein_iter)
            
            try:
                incontext_batch = next(incontext_iter)
            except StopIteration:
                incontext_iter = iter(self.dataset_incontext)
                incontext_batch = next(incontext_iter)
            
            try:
                mol_text_batch = next(mol_text_iter)
            except StopIteration:
                mol_text_iter = iter(self.dataset_mol_text)
                mol_text_batch = next(mol_text_iter)

            try:
                pro_text_batch = next(pro_text_iter)
            except StopIteration:
                pro_text_iter = iter(self.dataset_pro_text)
                pro_text_batch = next(pro_text_iter)    

            # Due to the multiple workers, the data in batch may be in random order
            yield text_batch, molecule_batch, protein_batch, incontext_batch, mol_text_batch, pro_text_batch



def get_model(args, config, tokenizer, logger):
    if args.model.checkpoint_path:
        model = T5ForConditionalGeneration(
            config,
        )
        model.resize_token_embeddings(len(tokenizer))
        model.load_state_dict(torch.load(args.model.checkpoint_path), strict=True)
        torch.cuda.empty_cache()
        logger.log_message(f"Loaded model from {args.model.checkpoint_path}")
    elif args.model.random_init:
        model = T5ForConditionalGeneration(
            config,
        )
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = T5ForConditionalGeneration.from_pretrained(
            args.model.name,
            config=config,
        )

    return model


def get_config(args):
    config = AutoConfig.from_pretrained(
        args.model.name,
    )
    config.dropout_rate = args.model.dropout
    return config


def get_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model.name,
        use_fast=True
    )
    tokenizer.model_max_length = int(1e9)

    amino_acids = [
        "A", "C", "D", "E", "F",
        "G", "H", "I", "K", "L",
        "M", "N", "P", "Q", "R",
        "S", "T", "V", "W", "Y"
    ]
    prefixed_amino_acids = [f"<p>{aa}" for aa in amino_acids]
    tokenizer.add_tokens(prefixed_amino_acids)
    
    selfies_dict_list = [line.strip() for line in open(os.path.join(__file__.split('biot5/utils')[0], args.molecule_dict))]
    tokenizer.add_tokens(selfies_dict_list)
    
    special_tokens_dict = {'additional_special_tokens': 
                           ['<bom>', '<eom>',
                           '<bop>', '<eop>',
                           'MOLECULE NAME', 'DESCRIPTION',
                           'PROTEIN NAME', 'FUNCTION', 'SUBCELLULAR LOCATION', 'PROTEIN FAMILIES']}
    tokenizer.add_special_tokens(special_tokens_dict, replace_additional_special_tokens=False)
    return tokenizer


def load_dataset_splits(args):
    if args.mode == 'pt':
        dataset_c4 = datasets.load_dataset(
            'c4',
            'en',
            streaming=True,
        )

        dataset_c4 = dataset_c4.remove_columns(
            ['timestamp', 'url']
        )

        dataset_splits_text = {
            'train': dataset_c4['train'],
            'test': dataset_c4['validation'],
        }

        dataset_zinc = datasets.load_dataset('zpn/zinc20', streaming=True)
        # dataset_zinc = dataset_zinc.remove_columns(['id', 'selfies'])
        # dataset_zinc = dataset_zinc.rename_column('smiles', 'text')
        dataset_zinc = dataset_zinc.remove_columns(['id', 'smiles'])
        dataset_zinc = dataset_zinc.rename_column('selfies', 'text')

        def molecule_process(sequence):
            return '<bom>' + sequence + '<eom>'
        # Prepend <p> to every protein sequence in the protein dataset
        dataset_zinc = dataset_zinc.map(lambda example: {'text': molecule_process(example['text'])})

        dataset_splits_molecule = {
            'train': dataset_zinc['train'],
            'test': dataset_zinc['validation'],
        }
        
        # Uniref90 with only 1 shards
        # dataset_uniref = datasets.load_dataset('zpn/uniref90', streaming=True, split='train')
        # dataset_uniref = dataset_uniref.remove_columns(['n', 'Tax', 'TaxID', 'RepID', 'description'])
        # dataset_uniref = dataset_uniref.rename_column('sequence', 'text')

        # Uniref50
        # dataset_uniref = datasets.load_dataset('zpn/uniref50', streaming=True, split='train')
        # dataset_uniref = dataset_uniref.remove_columns(['n', 'Tax', 'TaxID', 'RepID', '__index_level_0__'])
        # dataset_uniref = dataset_uniref.rename_column('sequence', 'text')
        # dataset_uniref['validation'] = dataset_uniref['train'].take(2_000_000)
        # dataset_uniref['train'] = dataset_uniref['train'].skip(20_000_000)
        dataset_uniref = datasets.load_dataset('text', data_files=
                                               {'train': [f"{args.pair_data_dir}/uniref50_2018_03.train.seqs.pro.nospace_{i+1}" for i in range(10)], 
                                                'test': [f"{args.pair_data_dir}/uniref50_2018_03.valid.seqs.pro.nospace"]}, streaming=True)
        
        def protein_process(sequence, character):
            return '<bop>' + ''.join([character + c for c in sequence]) + '<eop>'
        # Prepend <p> to every protein sequence in the protein dataset
        dataset_uniref = dataset_uniref.map(lambda example: {'text': protein_process(example['text'], '<p>')})

        # Uniref50 popular
        # dataset_uniref = datasets.load_dataset('agemagician/uniref50', streaming=True)
        # dataset_uniref = dataset_uniref.remove_columns(['id', 'name'])

        dataset_splits_protein = {
            'train': dataset_uniref['train'],
            'test': dataset_uniref['test'],
        }
        
        incontext_train_size = 1110
        dataset_incontext = datasets.load_dataset('text', data_files={'train': [f'{args.incontext_data_dir}/pubmed22n{"{:04}".format(i)}.txt' for i in range(1, incontext_train_size)], 
                                                                      'test': [f'{args.incontext_data_dir}/pubmed22n{"{:04}".format(i)}.txt' for i in range(incontext_train_size, 1115)]}, streaming=True)
        dataset_splits_incontext = {
            'train': dataset_incontext['train'],
            'test': dataset_incontext['test'],
        }

        dataset_mol_text = datasets.load_dataset('csv', data_files=[f'{args.pair_data_dir}/mol_text_nolap.tsv'], delimiter='\t')
        assert len(dataset_mol_text['train']) == 339422
        dataset_splits_mol_text = {
            'train': dataset_mol_text['train'],
            'test': dataset_mol_text['train'],
        }
        
        dataset_pro_text = datasets.load_dataset('csv', data_files=[f'{args.pair_data_dir}/pro_text.tsv'], delimiter='\t')
        assert len(dataset_pro_text['train']) == 569213
        dataset_splits_pro_text = {
            'train': dataset_pro_text['train'],
            'test': dataset_pro_text['train'],
        }
        # TODO check the time effect of n_shards or num_workers
        # assert (
        #     dataset_c4['train'].n_shards == 1024 and dataset_zinc['train'].n_shards == 1024 and dataset_uniref['train'].n_shards == 1024
        # ), "We want to have many shards for efficient processing with num_workes in PyTorch dataloader"
    elif args.mode == 'ft':
        dataset_splits = datasets.load_dataset(
            args.data.exec_file_path,
            data_dir=args.data.data_dir,
            task_dir=args.data.task_dir,
            max_num_instances_per_task=args.data.max_num_instances_per_task,
            max_num_instances_per_eval_task=args.data.max_num_instances_per_task
        )
        return dataset_splits
    else:
        raise NotImplementedError

    return dataset_splits_text, dataset_splits_molecule, dataset_splits_protein, dataset_splits_incontext, dataset_splits_mol_text, dataset_splits_pro_text


def process_dataset(dataset_splits, args, tokenizer):
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            # We increase the input_length, because instead of masking tokens T5 replaces
            # masked spans with a single token, therefore to avoid padding we need to have
            # longer sequences at the start, before masking
            before_mask_input_length, target_length = compute_input_and_target_lengths(
                inputs_length=args.data.input_length,
                noise_density=args.data.mlm_probability,
                mean_noise_span_length=args.data.mean_noise_span_length,
            )

            with open_dict(args):
                args.data.before_mask_input_length = before_mask_input_length
                args.data.target_length = target_length

            dataset_split = dataset_split.map(
                tokenize_function,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'in_length': before_mask_input_length,
                },
                remove_columns=['text'],
            )

            dataset_split = dataset_split.shuffle(buffer_size=10_000, seed=args.seed)
            final_datasets[split] = dataset_split
    elif args.mode == 'ft':
        final_datasets = dataset_splits
    else:
        raise NotImplementedError

    return final_datasets

def process_dataset_seq_desc(dataset_splits, args, tokenizer):
    if args.mode == 'pt':
        final_datasets = {}

        for split, dataset_split in dataset_splits.items():

            dataset_split = dataset_split.map(
                tokenize_function_seq_desc,
                batched=True,
                fn_kwargs={
                    'tokenizer': tokenizer,
                    'max_length': args.data.input_length,
                },
                remove_columns=['seq', 'desc'],
            )

            dataset_split = dataset_split.shuffle(seed=args.seed)
            final_datasets[split] = dataset_split
    else:
        raise NotImplementedError

    return final_datasets

def get_data_collator(tokenizer, config, args):
    if args.mode == 'pt':
        data_collator = DataCollatorForUnimptT5(
            tokenizer=tokenizer,
            noise_density=args.data.mlm_probability,
            mean_noise_span_length=args.data.mean_noise_span_length,
            input_length=args.data.input_length,
            target_length=args.data.target_length,
            pad_token_id=config.pad_token_id,
        )
    elif args.mode == 'ft':
        data_collator = DataCollatorForNI(
            tokenizer,
            padding="longest",
            max_source_length=args.data.max_seq_len,
            max_target_length=args.data.max_target_len,
            label_pad_token_id=-100,
            pad_to_multiple_of=8,
            add_task_name=args.data.add_task_name,
            add_task_definition=args.data.add_task_definition,
            num_pos_examples=args.data.num_pos_examples,
            num_neg_examples=args.data.num_neg_examples,
            add_explanation=args.data.add_explanation,
            tk_instruct=args.data.tk_instruct
        )
    else:
        raise NotImplementedError

    return data_collator


def get_dataloaders(tokenizer, config, args):
    if args.mode == 'pt':
        dataset_splits_text, dataset_splits_molecule, dataset_splits_protein, \
        dataset_splits_incontext, dataset_splits_mol_text, dataset_splits_pro_text = load_dataset_splits(args)
        dataset_text = process_dataset(dataset_splits=dataset_splits_text, args=args, tokenizer=tokenizer)
        dataset_molecule = process_dataset(dataset_splits=dataset_splits_molecule, args=args, tokenizer=tokenizer)
        dataset_protein = process_dataset(dataset_splits=dataset_splits_protein, args=args, tokenizer=tokenizer)
        dataset_incontext = process_dataset(dataset_splits=dataset_splits_incontext, args=args, tokenizer=tokenizer)
        dataset_mol_text = process_dataset_seq_desc(dataset_splits=dataset_splits_mol_text, args=args, tokenizer=tokenizer)
        dataset_pro_text = process_dataset_seq_desc(dataset_splits=dataset_splits_pro_text, args=args, tokenizer=tokenizer)
        data_collator = get_data_collator(tokenizer=tokenizer, config=config, args=args)

        is_iterable = isinstance(dataset_text['train'], IterableDataset) & isinstance(dataset_molecule['train'], IterableDataset) & isinstance(dataset_protein['train'], IterableDataset)

        dataloaders = {}

        for split in ['train', 'test']:
            batch_size = args.optim.batch_size // args.optim.grad_acc

            if split in ['test']:
                batch_size *= 2

            shuffle = (split == 'train') and not is_iterable

            if args.mode == 'ft' and split == 'train':
                assert shuffle is True
            else:
                assert shuffle is False

            mixed_dataset_split = MixedDataset(dataset_text[split], dataset_molecule[split], dataset_protein[split], dataset_incontext[split], dataset_mol_text[split], dataset_pro_text[split])
            dataloaders[split] = DataLoader(
                mixed_dataset_split,
                shuffle=shuffle,
                collate_fn=data_collator,
                batch_size=batch_size // 6,
                num_workers=args.data.num_workers,
                pin_memory=True,
                drop_last=False,
            )
    elif args.mode == 'ft':
        dataset_splits = load_dataset_splits(args)
        dataset = process_dataset(dataset_splits=dataset_splits, args=args, tokenizer=tokenizer)
        data_collator = get_data_collator(tokenizer=tokenizer, config=config,
                                        args=args)

        is_iterable = isinstance(dataset['train'], IterableDataset)

        dataloaders = {}

        for split in ['train', 'validation', 'test']:
            batch_size = args.optim.batch_size // args.optim.grad_acc

            if split in ['validation', 'test']:
                batch_size *= args.optim.test_bsz_multi

            shuffle = (split == 'train') and not is_iterable

            if args.mode == 'ft' and split == 'train':
                assert shuffle is True
            else:
                assert shuffle is False

            dataloaders[split] = DataLoader(
                dataset[split],
                shuffle=shuffle,
                collate_fn=data_collator,
                batch_size=batch_size,
                num_workers=args.data.num_workers,
                pin_memory=True,
                drop_last=False,
            )

    # Add & Check args about data loaders
    with open_dict(args):
        if not is_iterable:
            args.data.train_batches = len(dataloaders['train'])
            args.data.validation_batches = len(dataloaders['validation'])
            args.data.test_batches = len(dataloaders['test'])

        if args.optim.epochs > 0:
            assert not is_iterable
            args.optim.total_steps = len(dataloaders['train']) * args.optim.epochs // args.optim.grad_acc

        # We increase eval BS by 2, so decrease number of eval steps
        args.eval.corrected_steps = args.eval.steps / args.optim.grad_acc

    return dataloaders['train'], dataloaders['validation'], dataloaders['test']


def get_optimizer(model, args):
    no_decay = ["bias", "LayerNorm", "layernorm", "layer_norm", "ln"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.optim.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    if args.optim.name == 'adamw':
        from transformers import AdamW
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adamwscale':
        from .copied_utils import AdamWScale
        optimizer = AdamWScale(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
        )
    elif args.optim.name == 'adafactor':
        from transformers import Adafactor
        optimizer = Adafactor(
            optimizer_grouped_parameters,
            lr=args.optim.base_lr,
            relative_step=False,
        )
    else:
        raise NotImplementedError

    return optimizer


def get_lr_scheduler(optimizer, args, logger):
    if args.optim.lr_scheduler == 'cosine':
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            CosineAnnealingLR,
        )

        scheduler1 = LinearLR(
            optimizer,
            start_factor=0.5,
            end_factor=1,
            total_iters=args.optim.warmup_steps,
            last_epoch=-1,
        )

        scheduler2 = CosineAnnealingLR(
            optimizer,
            T_max=args.optim.total_steps - args.optim.warmup_steps,
            eta_min=args.optim.final_cosine,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[args.optim.warmup_steps]
        )
    elif args.optim.lr_scheduler == 'legacy':
        import math
        from torch.optim.lr_scheduler import (
            SequentialLR,
            LinearLR,
            LambdaLR,
        )

        msg = "You are using T5 legacy LR Schedule, it's independent from the optim.base_lr"
        logger.log_message(msg)

        num_steps_optimizer1 = math.ceil(args.optim.total_steps * 0.9)
        iters_left_for_optimizer2 = args.optim.total_steps - num_steps_optimizer1

        scheduler1 = LambdaLR(
            optimizer,
            lambda step: min(
                1e-2, 1.0 / math.sqrt(step)
            ) / args.optim.base_lr if step else 1e-2 / args.optim.base_lr
        )

        scheduler2 = LinearLR(
            optimizer,
            start_factor=(
                min(1e-2, 1.0 / math.sqrt(num_steps_optimizer1)) / args.optim.base_lr
            ),
            end_factor=0,
            total_iters=iters_left_for_optimizer2,
            last_epoch=-1,
        )

        lr_scheduler = SequentialLR(
            optimizer,
            schedulers=[scheduler1, scheduler2],
            milestones=[num_steps_optimizer1]
        )
    elif args.optim.lr_scheduler == 'constant':
        from transformers import get_scheduler
        lr_scheduler = get_scheduler(
            name=args.optim.lr_scheduler,
            optimizer=optimizer,
        )
    else:
        raise NotImplementedError

    return lr_scheduler
