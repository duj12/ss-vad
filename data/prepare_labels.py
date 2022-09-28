import numpy
import torch
import pandas as pd
import numpy as np
import argparse
from h5py import File
from pathlib import Path
from loguru import logger
import torch.utils.data as tdata
from tqdm import tqdm
from models import crnn, cnn10
import sys
import csv


class HDF5Dataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe. 
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self, h5file: File, transform=None):
        super(HDF5Dataset, self).__init__()
        self._h5file = h5file
        self.dataset = None
        # IF none is passed still use no transform at all
        self._transform = transform
        with File(self._h5file, 'r') as store:
            self._len = len(store)
            self._labels = list(store.keys())
            self.datadim = store[self._labels[0]].shape[-1]

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = File(self._h5file, 'r')
        fname = self._labels[index]
        data = self.dataset[fname][()]
        data = torch.as_tensor(data).float()
        if self._transform:
            data = self._transform(data)
        return data, fname

class MultiSingleHDF5Dataset(tdata.Dataset):
    """
    HDF5 dataset indexed by a labels dataframe.
    Indexing is done via the dataframe since we want to preserve some storage
    in cases where oversampling is needed ( pretty likely )
    """
    def __init__(self, fnames, h5files: numpy.ndarray, transform=None):
        super(MultiSingleHDF5Dataset, self).__init__()
        self._h5files = h5files
        self._len = len(h5files)
        self.dataset = None
        self._labels = fnames
        # IF none is passed still use no transform at all
        self._transform = transform
        # for _h5file in h5files:
        #     with File(_h5file, 'r') as store:
        #         key = list(store.keys())[0]
        #         self._labels.append(key)

    def __len__(self):
        return self._len

    def __getitem__(self, index):
        self.dataset = File(self._h5files[index], 'r')
        fname = self._labels[index]
        data = self.dataset[fname][()]
        data = torch.as_tensor(data).float()
        if self._transform:
            data = self._transform(data)
        return data, fname


MODELS = {
    'crnn': {
        'model': crnn,
        'encoder': torch.load('pretrained_models/labelencoders/teacher.pth'), #torch.load('encoders/balanced.pth'),
        'outputdim': 527,
    },
    'gpvb': {
        'model': crnn,
        'encoder': torch.load('pretrained_models/labelencoders/students.pth'), #torch.load('encoders/balanced_binary.pth'),
        'outputdim': 2,
    }
}

POOLING = {
    'max': lambda x: np.max(x, axis=-1),
    'mean': lambda x: np.mean(x, axis=-1)
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = torch.device(DEVICE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=Path)
    parser.add_argument('-m', '--model', default='crnn', type=str)
    parser.add_argument('-c', type=int, default=10)
    parser.add_argument('-po',
                        '--pool',
                        default='max',
                        choices=POOLING.keys(),
                        type=str)
    parser.add_argument('--pre', '-p', default='pretrained/gpv_f.pth')
    parser.add_argument('hdf5output', type=Path)
    parser.add_argument('csvoutput', type=Path)
    args = parser.parse_args()

    log_format = "[<green>{time:YYYY-MM-DD HH:mm:ss}</green>] {message}"
    logger.configure(handlers=[{"sink": sys.stderr, "format": log_format}])

    for k, v in vars(args).items():
        logger.info(f"{k} : {v}")

    model_dict = MODELS[args.model]
    global model
    model = model_dict['model'](outputdim=model_dict['outputdim'],
                                pretrained_from=args.pre).to(DEVICE).eval()
    encoder = model_dict['encoder']
    logger.info(model)
    pooling_fun = POOLING[args.pool]
    if Path(args.data).suffix == '.csv':
        data = pd.read_csv(args.data, sep='\s+')
        fnames = data['filename'].unique()
        data = data['hdf5path'].unique()
        #assert len(data) == 1, "Only single hdf5 supported yet"   # add mutiple hdf5 files support
        #data = data[0]
    else:  #h5 file directly
        #data = args.data
        raise NotImplementedError("Only supoort scv files like: filepath \t hdf5path")


    logger.info(f"Reading from {len(data)} input file")
    dataloader = tdata.DataLoader(MultiSingleHDF5Dataset(fnames, data),
                                   num_workers=4,
                                   batch_size=1)
    speech_class_idx = np.where(encoder.classes_ == 'Speech')[0]
    non_speech_idx = np.arange(len(encoder.classes_))
    non_speech_idx = np.delete(non_speech_idx, speech_class_idx)
    # with torch.no_grad(), File(args.hdf5output, 'w') as store, tqdm(
    #         total=len(dataloader)) as pbar, open(args.csvoutput,
    #                                              'w') as csvfile:
    #     abs_output_hdf5 = Path(args.hdf5output).absolute()
    #     csvwr = csv.writer(csvfile, delimiter='\t')
    #     csvwr.writerow(['filename', 'hdf5path'])
    #     for batch in dataloader:
    #         x, fname = batch
    #         fname = fname[0]
    #         x = x.to(DEVICE)
    #         if x.shape[1] < 8:
    #             continue
    #         clip_pred, time_pred = model(x)
    #         clip_pred = clip_pred.squeeze(0).to('cpu').numpy()
    #         time_pred = time_pred.squeeze(0).to('cpu').numpy()
    #         speech_time_pred = time_pred[..., speech_class_idx].squeeze(-1)
    #         speech_clip_pred = clip_pred[..., speech_class_idx].squeeze(-1)
    #         non_speech_clip_pred = clip_pred[..., non_speech_idx]
    #         non_speech_time_pred = time_pred[..., non_speech_idx]
    #         non_speech_time_pred = pooling_fun(non_speech_time_pred)
    #         store[f'{fname}/speech'] = speech_time_pred
    #         store[f'{fname}/noise'] = non_speech_time_pred
    #         store[f'{fname}/clipspeech'] = speech_clip_pred
    #         store[f'{fname}/clipnoise'] = non_speech_clip_pred
    #         csvwr.writerow([fname, abs_output_hdf5])
    #         pbar.set_postfix(fname=fname, speechsize=speech_time_pred.shape)
    #         pbar.update()

    with torch.no_grad(), open(args.csvoutput,'w') as csvfile, tqdm(total=len(data)) as pbar:
        abs_output_hdf5 = Path(args.hdf5output).absolute()
        csvwr = csv.writer(csvfile, delimiter='\t')
        csvwr.writerow(['filename', 'hdf5path'])

        # #尝试使用process.map进行多线程并行
        # from pypeln import process as pr
        # def forward_label(hdf5path):
        #     dataset = File(hdf5path, 'r')
        #     fname = list(dataset.keys())[0]
        #     data = dataset[fname][()]
        #     x = torch.as_tensor(data).float()
        #     if x.shape[1] < 8:
        #         return None, None
        #     clip_pred, time_pred = model(x)  #这里在CPU的多线程里面调用模型前向推理会出问题
        #     return clip_pred, time_pred, fname
        # model = model.to('cpu')   #process.map 多线程并行只能在cpu上进行
        # for clip_pred, time_pred, fname in tqdm(pr.map(forward_label,
        #                                         data,
        #                                         workers=args.c, maxsize=4),
        #                         total=len(dataloader)):
        #     if clip_pred == None:
        #         continue

        for batch in dataloader:
            x, fname = batch
            fname = fname[0]
            x = x.to(DEVICE)
            if x.shape[1] < 8:
                continue
            clip_pred, time_pred = model(x)

            clip_pred = clip_pred.squeeze(0).to('cpu').numpy()
            time_pred = time_pred.squeeze(0).to('cpu').numpy()
            speech_time_pred = time_pred[..., speech_class_idx].squeeze(-1)
            speech_clip_pred = clip_pred[..., speech_class_idx].squeeze(-1)
            non_speech_clip_pred = clip_pred[..., non_speech_idx]
            non_speech_time_pred = time_pred[..., non_speech_idx]
            non_speech_time_pred = pooling_fun(non_speech_time_pred)
            with File(f'{args.hdf5output}/{fname}.h5', 'w') as store:
                store[f'{fname}/speech'] = speech_time_pred
                store[f'{fname}/noise'] = non_speech_time_pred
                store[f'{fname}/clipspeech'] = speech_clip_pred
                store[f'{fname}/clipnoise'] = non_speech_clip_pred
            csvwr.writerow([fname, f'{args.hdf5output}/{fname}.h5'])
            pbar.set_postfix(fname=fname, speechsize=speech_time_pred.shape)
            pbar.update()


if __name__ == "__main__":
    main()
