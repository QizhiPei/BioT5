import math
import torch
from torch.utils.data import get_worker_info
from datasets.iterable_dataset import IterableDataset

class MixedDataset(IterableDataset):
    def __init__(self, dataset_text, dataset_molecule, dataset_protein, dataset_incontext, dataset_mol_text, dataset_pro_text, dataset_pubmed_text, dataset_name_selfies, split='train'):
        super().__init__(self, split=split)
        if split == 'train':
            self.dataset_text = dataset_text
            self.dataset_molecule = dataset_molecule
            self.dataset_protein = dataset_protein
            self.dataset_incontext = dataset_incontext
            self.dataset_mol_text = dataset_mol_text
            self.dataset_pro_text = dataset_pro_text
            self.dataset_pubmed_text = dataset_pubmed_text
            self.dataset_name_selfies = dataset_name_selfies
        elif split == 'test':
            self.dataset_mol_text = dataset_mol_text
        else:
            raise NotImplementedError
    
    def __iter__(self):
        if self.split == 'train':
            worker_info = get_worker_info()
            if worker_info is None:
                text_iter = iter(self.dataset_text)
                molecule_iter = iter(self.dataset_molecule)
                protein_iter = iter(self.dataset_protein)
                incontext_iter = iter(self.dataset_incontext)
                mol_text_iter = iter(self.dataset_mol_text)
                pro_text_iter = iter(self.dataset_pro_text)
                pubmed_text_iter = iter(self.dataset_pubmed_text)
                name_selfies_iter = iter(self.dataset_name_selfies)

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

                    try:
                        pubmed_text_batch = next(pubmed_text_iter)
                    except StopIteration:
                        pubmed_text_iter = iter(self.dataset_pubmed_text)
                        pubmed_text_batch = next(pubmed_text_iter) 
                    
                    try: 
                        name_selfies_batch = next(name_selfies_iter)
                    except StopIteration:
                        name_selfies_iter = iter(self.dataset_name_selfies)
                        name_selfies_batch = next(name_selfies_iter)

                    # Due to the multiple workers, the data in batch may be in random order
                    yield text_batch, molecule_batch, protein_batch, incontext_batch, mol_text_batch, pro_text_batch, pubmed_text_batch, name_selfies_batch
            else:
                worker_id = worker_info.id
                if worker_id % 8 == 0:
                    text_iter = iter(self.dataset_text)
                    while True:
                        try:
                            text_batch = next(text_iter)
                        except StopIteration:
                            text_iter = iter(self.dataset_text)
                            text_batch = next(text_iter)
                        yield text_batch
                elif worker_id % 8 == 1:
                    molecule_iter = iter(self.dataset_molecule)
                    while True:
                        try:
                            molecule_batch = next(molecule_iter)
                        except StopIteration:
                            molecule_iter = iter(self.dataset_molecule)
                            molecule_batch = next(molecule_iter)
                        yield molecule_batch
                elif worker_id % 8 == 2:
                    protein_iter = iter(self.dataset_protein)
                    while True:
                        try:
                            protein_batch = next(protein_iter)
                        except StopIteration:
                            protein_iter = iter(self.dataset_protein)
                            protein_batch = next(protein_iter)
                        yield protein_batch
                elif worker_id % 8 == 3:
                    incontext_iter = iter(self.dataset_incontext)
                    while True:
                        try:
                            incontext_batch = next(incontext_iter)
                        except StopIteration:
                            incontext_iter = iter(self.dataset_incontext)
                            incontext_batch = next(incontext_iter)
                        yield incontext_batch
                elif worker_id % 8 == 4:
                    mol_text_iter = iter(self.dataset_mol_text)
                    while True:
                        try:
                            mol_text_batch = next(mol_text_iter)
                        except StopIteration:
                            mol_text_iter = iter(self.dataset_mol_text)
                            mol_text_batch = next(mol_text_iter)
                        yield mol_text_batch
                elif worker_id % 8 == 5:
                    pro_text_iter = iter(self.dataset_pro_text)
                    while True:
                        try:
                            pro_text_batch = next(pro_text_iter)
                        except StopIteration:
                            pro_text_iter = iter(self.dataset_pro_text)
                            pro_text_batch = next(pro_text_iter)
                        yield pro_text_batch
                elif worker_id % 8 == 6:
                    pubmed_text_iter = iter(self.dataset_pubmed_text)
                    while True:
                        try:
                            pubmed_text_batch = next(pubmed_text_iter)
                        except StopIteration:
                            pubmed_text_iter = iter(self.dataset_pubmed_text)
                            pubmed_text_batch = next(pubmed_text_iter)
                        yield pubmed_text_batch
                elif worker_id % 8 == 7:
                    name_selfies_iter = iter(self.dataset_name_selfies)
                    while True:
                        try:
                            name_selfies_batch = next(name_selfies_iter)
                        except StopIteration:
                            name_selfies_iter = iter(self.dataset_name_selfies)
                            name_selfies_batch = next(name_selfies_iter)
                        yield name_selfies_batch
                    
        elif self.split == 'test':
            mol_text_start = 0
            mol_text_end = len(self.dataset_mol_text)
            worker_info = get_worker_info()
            if worker_info is None:  # single-process data loading, return the full iterator
                iter_start = mol_text_start
                iter_end = mol_text_end
            else:  # in a worker process
                # split workload
                per_worker = int(math.ceil((mol_text_end - mol_text_start) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = mol_text_start + worker_id * per_worker
                iter_end = min(iter_start + per_worker, mol_text_end)
            mol_text_iter = iter(self.dataset_mol_text.select(range(iter_start,iter_end)))
            for mol_text_batch in mol_text_iter:
                yield mol_text_batch

        else:
            raise NotImplementedError

class MixedDataset_Abl(IterableDataset):
    def __init__(self, dataset_text, dataset_molecule, dataset_protein, dataset_incontext, dataset_mol_text, dataset_pro_text, dataset_name_selfies, split='train'):
        super().__init__(self, split=split)
        if split == 'train':
            self.dataset_text = dataset_text
            self.dataset_molecule = dataset_molecule
            self.dataset_protein = dataset_protein
            self.dataset_incontext = dataset_incontext
            self.dataset_mol_text = dataset_mol_text
            self.dataset_pro_text = dataset_pro_text
            self.dataset_name_selfies = dataset_name_selfies
        elif split == 'test':
            self.dataset_mol_text = dataset_mol_text
        else:
            raise NotImplementedError

    def __iter__(self):
        if self.split == 'train':
            worker_info = get_worker_info()
            if worker_info is None:
                text_iter = iter(self.dataset_text)
                molecule_iter = iter(self.dataset_molecule)
                protein_iter = iter(self.dataset_protein)
                incontext_iter = iter(self.dataset_incontext)
                mol_text_iter = iter(self.dataset_mol_text)
                pro_text_iter = iter(self.dataset_pro_text)
                name_selfies_iter = iter(self.dataset_name_selfies)

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
                    
                    try: 
                        name_selfies_batch = next(name_selfies_iter)
                    except StopIteration:
                        name_selfies_iter = iter(self.dataset_name_selfies)
                        name_selfies_batch = next(name_selfies_iter)

                    # Due to the multi-workers, the data in batch may be in random order
                    yield text_batch, molecule_batch, protein_batch, incontext_batch, mol_text_batch, pro_text_batch, name_selfies_batch

            else:
                worker_id = worker_info.id
                if worker_id % 7 == 0:
                    text_iter = iter(self.dataset_text)
                    while True:
                        try:
                            text_batch = next(text_iter)
                        except StopIteration:
                            text_iter = iter(self.dataset_text)
                            text_batch = next(text_iter)
                        yield text_batch
                elif worker_id % 7 == 1:
                    molecule_iter = iter(self.dataset_molecule)
                    while True:
                        try:
                            molecule_batch = next(molecule_iter)
                        except StopIteration:
                            molecule_iter = iter(self.dataset_molecule)
                            molecule_batch = next(molecule_iter)
                        yield molecule_batch
                elif worker_id % 7 == 2:
                    protein_iter = iter(self.dataset_protein)
                    while True:
                        try:
                            protein_batch = next(protein_iter)
                        except StopIteration:
                            protein_iter = iter(self.dataset_protein)
                            protein_batch = next(protein_iter)
                        yield protein_batch
                elif worker_id % 7 == 3:
                    incontext_iter = iter(self.dataset_incontext)
                    while True:
                        try:
                            incontext_batch = next(incontext_iter)
                        except StopIteration:
                            incontext_iter = iter(self.dataset_incontext)
                            incontext_batch = next(incontext_iter)
                        yield incontext_batch
                elif worker_id % 7 == 4:
                    mol_text_iter = iter(self.dataset_mol_text)
                    while True:
                        try:
                            mol_text_batch = next(mol_text_iter)
                        except StopIteration:
                            mol_text_iter = iter(self.dataset_mol_text)
                            mol_text_batch = next(mol_text_iter)
                        yield mol_text_batch
                elif worker_id % 7 == 5:
                    pro_text_iter = iter(self.dataset_pro_text)
                    while True:
                        try:
                            pro_text_batch = next(pro_text_iter)
                        except StopIteration:
                            pro_text_iter = iter(self.dataset_pro_text)
                            pro_text_batch = next(pro_text_iter)
                        yield pro_text_batch
                elif worker_id % 7 == 6:
                    name_selfies_iter = iter(self.dataset_name_selfies)
                    while True:
                        try:
                            name_selfies_batch = next(name_selfies_iter)
                        except StopIteration:
                            name_selfies_iter = iter(self.dataset_name_selfies)
                            name_selfies_batch = next(name_selfies_iter)
                        yield name_selfies_batch

        elif self.split == 'test':
            mol_text_start = 0
            mol_text_end = len(self.dataset_mol_text)
            worker_info = get_worker_info()
            if worker_info is None:  # single-process data loading, return the full iterator
                iter_start = mol_text_start
                iter_end = mol_text_end
            else:  # in a worker process
                # split workload
                per_worker = int(math.ceil((mol_text_end - mol_text_start) / float(worker_info.num_workers)))
                worker_id = worker_info.id
                iter_start = mol_text_start + worker_id * per_worker
                iter_end = min(iter_start + per_worker, mol_text_end)
            mol_text_iter = iter(self.dataset_mol_text.select(range(iter_start,iter_end)))
            for mol_text_batch in mol_text_iter:
                yield mol_text_batch

        else:
            raise NotImplementedError