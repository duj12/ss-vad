import torch
import sys
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import utils
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import uuid
import argparse
from models import crnn
import os
import matplotlib.pyplot as plt

SAMPLE_RATE = 16000 #22050
EPS = np.spacing(1)
#开源模型帧率为50
LMS_ARGS = {
    'n_fft': 2048,
    'n_mels': 64,
    'hop_length': int(SAMPLE_RATE * 0.02),
    'win_length': int(SAMPLE_RATE * 0.04)
}

#新版模型将帧率改成了80
LMS_ARGS_new = {
    'n_fft': 2048,
    'n_mels': 64,
    'hop_length': int(SAMPLE_RATE * 0.0125),
    'win_length': int(SAMPLE_RATE * 0.025)
}

DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 'cuda'
DEVICE = torch.device(DEVICE)


def extract_feature(wavefilepath, **kwargs):
    _, file_extension = os.path.splitext(wavefilepath)
    if file_extension == '.wav':
        #wav, sr = sf.read(wavefilepath, dtype='float32')
        wav, sr = librosa.read(wavefilepath)   #librosa和soundfile读取得到的数值有细微的差别
    if file_extension == '.mp3':
        wav, sr = librosa.load(wavefilepath)
    elif file_extension not in ['.mp3', '.wav']:
        raise NotImplementedError('Audio extension not supported... yet ;)')
    if wav.ndim > 1:
        wav = wav.mean(-1)
    wav = librosa.resample(wav, sr, target_sr=SAMPLE_RATE)
    return np.log(
        librosa.feature.melspectrogram(wav.astype(np.float32), SAMPLE_RATE, **
                                       kwargs) + EPS).T


class OnlineLogMelDataset(torch.utils.data.Dataset):
    def __init__(self, data_list, **kwargs):
        super().__init__()
        self.dlist = data_list
        self.kwargs = kwargs

    def __getitem__(self, idx):
        return extract_feature(wavefilepath=self.dlist[idx],
                               **self.kwargs), self.dlist[idx]

    def __len__(self):
        return len(self.dlist)


MODELS = {
    'xmov': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'xmov/hard.pt',
        'resolution': 0.0125
    },
    'xmov1': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'xmov/hard+soft.pt',
        'resolution': 0.0125
    },
    'xmov0': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'xmov/hard+soft+clip.pt',
        'resolution': 0.0125
    },
    'xmov-s': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'xmov/stream-hard.pt',
        'resolution': 0.0125,
        'gru_bidirection': False
    },
    'xmov-s1': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'xmov/stream-hard+clip.pt',
        'resolution': 0.0125,
        'gru_bidirection': False
    },
    't1': {
        'model': crnn,
        'outputdim': 527,
        'encoder': 'labelencoders/teacher.pth',
        'pretrained': 'teacher1/model.pth',
        'resolution': 0.02
    },
    't2': {
        'model': crnn,
        'outputdim': 527,
        'encoder': 'labelencoders/teacher.pth',
        'pretrained': 'teacher2/model.pth',
        'resolution': 0.02
    },
    'sre': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'sre/model.pth',
        'resolution': 0.02
    },
    'v2': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'vox2/model.pth',
        'resolution': 0.02
    },
    'a2': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'audioset2/model.pth',
        'resolution': 0.02
    },
    'a2_v2': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'audio2_vox2/model.pth',
        'resolution': 0.02
    },
    'c1': {
        'model': crnn,
        'outputdim': 2,
        'encoder': 'labelencoders/students.pth',
        'pretrained': 'c1/model.pth',
        'resolution': 0.02
    },
}
def plot_wav_and_vad(audio, ref, hyp, save_name):
    '''
    plot wave figure and VAD reference and hypothesis
    :param audio:
    :param ref:
    :param hyp:
    :return:
    '''
    plt.figure()
    time1 = np.arange(0, len(hyp))
    time = np.arange(0, len(audio)) * (len(hyp) / len(audio))  #将时间轴映射到标签的数量
    plt.plot(time, audio)
    if not ref is None:
        assert(len(hyp)==len(ref))
        plt.plot(time1, ref)
    plt.plot(time1, hyp)
    #plt.show()
    plt.savefig(save_name)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '-w',
        '--wav',
        help=
        'A single wave/mp3/flac or any other compatible audio file with soundfile.read'
    )
    group.add_argument(
        '-l',
        '--wavlist',
        help=
        'A list of wave or any other compatible audio files. E.g., output of find . -type f -name *.wav > wavlist.txt'
    )
    parser.add_argument(
        '-r',
        '--reference',
        help='the frame wise VAD reference file.'
    )
    parser.add_argument('-model', choices=list(MODELS.keys()), default='sre')
    parser.add_argument(
        '--pretrained_dir',
        default='pretrained_models',
        help=
        'Path to downloaded pretrained models directory, (default %(default)s)'
    )
    parser.add_argument('-o',
                        '--output_path',
                        default=None,
                        help='Output folder to save predictions if necessary')
    parser.add_argument('--fig_save_path',
                        default=None,
                        help='Output folder to save wave and VAD result figures')
    parser.add_argument('-soft',
                        default=False,
                        action='store_true',
                        help='Outputs soft probabilities.')
    parser.add_argument('-hard',
                        default=False,
                        action='store_true',
                        help='Outputs hard labels as zero-one array.')
    parser.add_argument('-th',
                        '--threshold',
                        default=(0.5, 0.1),
                        type=float,
                        nargs="+")
    args = parser.parse_args()
    pretrained_dir = Path(args.pretrained_dir)
    if not (pretrained_dir.exists() and pretrained_dir.is_dir()):
        logger.error(f"""Pretrained directory {args.pretrained_dir} not found.
Please download the pretrained models from and try again or set --pretrained_dir to your directory."""
                     )
        return
    logger.info("Passed args")
    for k, v in vars(args).items():
        logger.info(f"{k} : {str(v):<10}")
    if args.wavlist:
        wavlist = pd.read_csv(args.wavlist,
                              usecols=[0],
                              header=None,
                              names=['filename'])
        wavlist = wavlist['filename'].values.tolist()
    elif args.wav:
        wavlist = [args.wav]
    #dset = OnlineLogMelDataset(wavlist, **LMS_ARGS)

    dset = OnlineLogMelDataset(wavlist, **LMS_ARGS_new)
    dloader = torch.utils.data.DataLoader(dset,
                                          batch_size=1,
                                          num_workers=3,
                                          shuffle=False)

    model_kwargs_pack = MODELS[args.model]
    model_resolution = model_kwargs_pack['resolution']
    # Load model from relative path
    model = model_kwargs_pack['model'](
        outputdim=model_kwargs_pack['outputdim'],
        pretrained_from=pretrained_dir /
        model_kwargs_pack['pretrained'],
        gru_bidirection=model_kwargs_pack.get('gru_bidirection', True)
    ).to(DEVICE).eval()
    encoder = torch.load(pretrained_dir / model_kwargs_pack['encoder'])
    logger.trace(model)

    output_dfs = []
    frame_outputs = {}
    frame_outputs_soft = []
    frame_outputs_hard = []
    threshold = tuple(args.threshold)

    speech_label_idx = np.where('Speech' == encoder.classes_)[0].squeeze()
    # Using only binary thresholding without filter
    if len(threshold) == 1:
        postprocessing_method = utils.binarize
    else:
        postprocessing_method = utils.double_threshold
    with torch.no_grad(), tqdm(total=len(dloader), leave=False,
                               unit='clip') as pbar:
        # 提供了参考标注，以及预测hard标签
        if args.reference and args.hard and args.soft:
            reference = []
            fin = open(args.reference, 'r')
            speech_soft_pred = 0
            speech_hard_pred = 0
        for i, (feature,filename) in enumerate(dloader):
            feature = torch.as_tensor(feature).to(DEVICE)
            prediction_tag, prediction_time = model(feature)
            prediction_tag = prediction_tag.to('cpu')
            prediction_time = prediction_time.to('cpu')

            if prediction_time is not None:  # Some models do not predict timestamps

                cur_filename = filename[0]  #Remove batchsize
                thresholded_prediction = postprocessing_method(
                    prediction_time, *threshold)
                speech_soft_pred = prediction_time[..., speech_label_idx]
                if args.soft:
                    speech_soft_pred = prediction_time[
                        ..., speech_label_idx].numpy()
                    frame_outputs[cur_filename] = speech_soft_pred[
                        0]  # 1 batch
                    frame_outputs_soft.append(speech_soft_pred[0])
                if args.hard:
                    speech_hard_pred = thresholded_prediction[...,
                                                              speech_label_idx]
                    frame_outputs[cur_filename] = speech_hard_pred[
                        0]  # 1 batch
                    frame_outputs_hard.append(speech_hard_pred[0])

                labelled_predictions = utils.decode_with_timestamps(
                    encoder, thresholded_prediction)
                pred_label_df = pd.DataFrame(
                    labelled_predictions[0],
                    columns=['event_label', 'onset', 'offset'])
                if not pred_label_df.empty:
                    pred_label_df['filename'] = cur_filename
                    pred_label_df['onset'] *= model_resolution
                    pred_label_df['offset'] *= model_resolution
                    pbar.set_postfix(labels=','.join(
                        np.unique(pred_label_df['event_label'].values)))
                    pbar.update()
                    output_dfs.append(pred_label_df)
            # 提供了参考标注，以及预测hard标签
            if args.reference and args.hard and args.soft:
                line = fin.readline()
                hyp = speech_hard_pred[0] #[0] remove batch
                ref = np.array([int(x) for x in line.strip().split(' ')])
                hyp_sz = (hyp.shape)[0]
                ref_sz = (ref.shape)[0]
                ref2hyp_ratio = float(ref_sz / hyp_sz)
                if (hyp_sz != ref_sz):  # 对标签进行采样
                    target_inds = (np.arange(hyp_sz) * ref2hyp_ratio).astype(int)
                    # 这里为了避免对齐之后的标签越界，将超过标签长度的idx替换成最后一个标签序号
                    target_inds[target_inds >= ref_sz] = ref_sz - 1
                    ref = ref[target_inds]
                assert (hyp.size == ref.size)
                reference.append(ref)
                wav_name = cur_filename.split('/')[-1].replace(".wav","")
                audio = librosa.load(cur_filename, sr=SAMPLE_RATE)[0]
                hypothesis = speech_soft_pred[0]
                plot_wav_and_vad(audio, ref, hypothesis, f"{args.fig_save_path}/{wav_name}.png")
            else:
                #只画出模型输出的结果
                wav_name = cur_filename.split('/')[-1].replace(".wav", "")
                audio = librosa.load(cur_filename, sr=SAMPLE_RATE)[0]
                hypothesis = speech_soft_pred[0]
                plot_wav_and_vad(audio, None, hypothesis, f"{args.fig_save_path}/{wav_name}.png")

    if args.reference and args.hard and args.soft:
        assert (len(reference) == len(frame_outputs_hard))
        logger.info("Calculating VAD measures ... ")
        import metrics
        speech_frame_ground_truth = np.concatenate(reference,
                                                   axis=0)
        speech_frame_predictions = np.concatenate(frame_outputs_hard,
                                                  axis=0)
        speech_frame_prob_predictions = np.concatenate(frame_outputs_soft,
                                                       axis=0)

        vad_results = []
        tn, fp, fn, tp = metrics.confusion_matrix(
            speech_frame_ground_truth, speech_frame_predictions).ravel()
        fer = 100 * ((fp + fn) / len(speech_frame_ground_truth))
        acc = 100 * ((tp + tn) / (len(speech_frame_ground_truth)))

        p_miss = 100 * (fn / (fn + tp))
        p_fa = 100 * (fp / (fp + tn))
        for i in [0.1, 0.3, 0.5, 0.7, 0.9]:
            mp_fa, mp_miss = metrics.obtain_error_rates(
                speech_frame_ground_truth, speech_frame_prob_predictions, i)
            tn, fp, fn, tp = metrics.confusion_matrix(
                speech_frame_ground_truth,
                speech_frame_prob_predictions > i).ravel()
            sub_fer = 100 * ((fp + fn) / len(speech_frame_ground_truth))
            logger.info(
                f"PFa {100 * mp_fa:.2f} Pmiss {100 * mp_miss:.2f} FER {sub_fer:.2f} t: {i:.2f}"
            )

        auc = metrics.roc(speech_frame_ground_truth,
                          speech_frame_prob_predictions) * 100
        for avgtype in ('micro', 'macro', 'binary'):
            precision, recall, f1, _ = metrics.precision_recall_fscore_support(
                speech_frame_ground_truth,
                speech_frame_predictions,
                average=avgtype)
            vad_results.append(
                (avgtype, 100 * precision, 100 * recall, 100 * f1))
        for avgtype, precision, recall, f1 in vad_results:
            print(
                f"VAD {avgtype:<10} F1: {f1:<10.3f} Pre: {precision:<10.3f} Recall: {recall:<10.3f}"
            )
        print(f"FER: {fer:.2f}")
        print(f"AUC: {auc:.2f}")
        print(f"Pfa: {p_fa:.2f}")
        print(f"Pmiss: {p_miss:.2f}")
        print(f"ACC: {acc:.2f}")

    full_prediction_df = pd.concat(output_dfs).sort_values(by='onset',ascending=True).reset_index()
    prediction_df = full_prediction_df[full_prediction_df['event_label'] ==
                                       'Speech']
    if args.output_path:
        args.output_path = Path(args.output_path)
        args.output_path.mkdir(parents=True, exist_ok=True)
        prediction_df.to_csv(args.output_path / 'speech_predictions.tsv',
                             sep='\t',
                             index=False)
        full_prediction_df.to_csv(args.output_path / 'all_predictions.tsv',
                                  sep='\t',
                                  index=False)

        if args.soft or args.hard:
            prefix = 'soft' if args.soft else 'hard'
            with open(args.output_path / f'{prefix}_predictions.txt',
                      'w') as wp:
                np.set_printoptions(suppress=True,
                                    precision=2,
                                    linewidth=np.inf)
                for fname, output in frame_outputs.items():
                    print(f"{fname} {output}", file=wp)
        logger.info(f"Putting results also to dir {args.output_path}")
    # if args.soft or args.hard:
    #     np.set_printoptions(suppress=True, precision=2, linewidth=np.inf)
    #     for fname, output in frame_outputs.items():
    #         print(f"{fname} {output}")
    # else:
    #     print(prediction_df.to_markdown(showindex=False))





if __name__ == "__main__":
    main()
