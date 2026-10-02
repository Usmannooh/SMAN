import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["PYTHONHASHSEED"] = "0"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import pickle
import random
import numpy as np
import torch
from models.base_iu import BASE_IU
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.tokenizers import Tokenizer
from modules.trainer import Trainer
from modules.loss import compute_loss

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default=r'E:\usman\DATA_SETS\data\iu_xray\images', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='./data/iu_xray/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--vocab_path', type=str, default='./data/New_vocab.pkl',help='Path to vocab file ')
    parser.add_argument('--embedding_path', type=str, default='./data/embedding/embeddings.txt', help='Path to text embeddings file ')
    parser.add_argument('--num_classes', type=int, default=31, help='')

    parser.add_argument('--save_dir', type=str, default='./records/', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='./records',help='the patch to save the results of experiments.')

    # mimic-cxr
    # parser.add_argument('--image_dir', type=str, default=r'E:\usman\DATA_SETS\data\mimic_cxr\images',  help='the path to the directory containing the data.')
    # parser.add_argument('--ann_path', type=str, default=r'E:\usman\DATA_SETS\data\mimic_cxr\annotation.json', help='the path to the directory containing the data.')
    # parser.add_argument('--vocab-path', type=str, default=r'E:\usman\DATA_SETS\ata\New_vocab.pkl')
    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--threshold', type=int, default=1, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=0, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch') 

    # Multimodal Decoder
    parser.add_argument('--m2_max_seq_length', type=int, default=60,help='the maximum sequence length of the reports.maximum:100')
    parser.add_argument('--kd', type=int, default=768, help='Tag feature medical embeddings size')
    parser.add_argument('--ssl_loss_weight', type=float, default=0.01,  #0.01,0.03,0.05,0.07,0.1
                        help='weighting coefficient for auxiliary loss for  Relational Contrastive Learning(default: 0.01)')
    parser.add_argument('--m2_decoder_weight', type=float, default=5e-5, help='weighting coefficient for M2Decoder loss (default: 0.05)') 
    parser.add_argument('--m2_decoder_seq_len', type=int, default=4, help='Length of decoder input sequence')
    parser.add_argument('--m2_decoder_target_start', type=int, default=1,help='Start index for target tokens')
    # Multimodal Decoder Projections
    parser.add_argument('--m2_proj_fc_in_dim', type=int, default=4096, help='Input dimension for m2_proj_fc linear layer ')
    parser.add_argument('--m2_proj_att_in_dim', type=int, default=2048, help='Input dimension for m2_proj_att linear layer ')
    parser.add_argument('--m2_proj_out_dim', type=int, default=768, help='Output dimension for m2_proj_fc/m2_proj_att linear layers')
    parser.add_argument('--max_seq_length', type=int, default=50, help='the maximum sequence length of the reports .')
    parser.add_argument('--fusion_strategy', type=str, default='concat_raw', choices=['concat_avg', 'average', 'concat_raw'], help='Multi-view feature fusion strategy')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,  help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.3, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    parser.add_argument('--hidden_dim', type=int, default=768, help='Hidden dimension size for adaptive gating')
    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the number of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.') 
    parser.add_argument('--early_stop', type=int, default=40, help='the patience of training.')
    parser.add_argument('--log_period', type=int, default=200, help='the logging interval (in batches).')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=0.002, help='the learning rate for the visual extractor.') 
    parser.add_argument('--lr_ed', type=float, default=7e-4, help='the Transformer(text) learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=0.00005,  help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98),  help='betas for Adam optimizer (momentum and variance tracking).')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='Epsilon value for the Adam optimizer for numerical stability.(Gradients vanishing)')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=4000,help='Zyada warmup steps se model ko initial training phase mein zyada stability milti hai')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')
    parser.add_argument('--momentum', type=float, default=0.9,  help='Momentum factor for SGD optimizer.')
    parser.add_argument('--nesterov', type=bool, default=True, help='Use Nesterov momentum in SGD optimizer.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=20, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.5, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='Set seed for reproducibility of results.') 
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    #parser.add_argument('--resume', type=str, default= r'E:\usman\SMAN.pth', help='Path to resume the training from existing checkpoint.')

    args = parser.parse_args()
    return args

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except:
        torch.use_deterministic_algorithms( False)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32 # Possible float32 ops ka rounding difference to other machine.
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def main():
    args = parse_agrs()
    seed_everything(args.seed)

    import pickle
    class RestrictedUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            # Allow only safe classes
            allowed_modules = {'builtins', '__main__', 'collections', 'torch', 'numpy'}
            if module in allowed_modules and '__' not in name:
                return super().find_class(module, name)
            # Forbid everything else
            raise pickle.UnpicklingError(f"Unsafe unpickling: {module}.{name}")

    with open(args.vocab_path, 'rb') as f:
        vocab = RestrictedUnpickler(f).load()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tokenizer = Tokenizer(args, device=device)#, device=device

    # DETERM GENERATOR For DataLoader
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True, generator=g, worker_init_fn=seed_worker)
    val_dataloader = R2DataLoader( args, tokenizer, split='val', shuffle=False,generator=g, worker_init_fn=seed_worker)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False,generator=g, worker_init_fn=seed_worker )

    #ye as lye Add he qk mimic mein add thi.
    with open(r'E:\usman\GOU1\SMAN_X_SHARE\data\embed.txt', 'r') as matrix_file:
        adjacency_matrix = [[int(num) for num in line.split(',')] for line in matrix_file]
    forward_adj = torch.tensor(adjacency_matrix, dtype=torch.float).to('cuda:0')
    identity_matrix = torch.eye(args.num_classes).to('cuda:0')
    backward_adj = forward_adj.t().to('cuda:0')
    forward_adj = forward_adj.add(identity_matrix).to('cuda:0')
    backward_adj = backward_adj.add(identity_matrix).to('cuda:0')
    model = BASE_IU(args, tokenizer, args.num_classes, forward_adj, backward_adj)


    criterion = compute_loss
    metrics = compute_scores
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)

    trainer.train()



if __name__ == '__main__':
    main()
