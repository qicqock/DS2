import os, sys
import argparse
import pytorch_lightning as pl

from ds2.pftune_models.pftune_model import PrefixSummarizationModule

def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser = pl.Trainer.add_argparse_args(parser)
    
    # ds2 dependent
    
    parser.add_argument("--exp_name", type=str, required=True, help="exp name for logging")
    parser.add_argument("--model_checkpoint", type=str, default="t5-large", help="Path, url or short name of the model") # roles "output_dir"
    parser.add_argument("--state_converter", type=str, default="mwz", choices=["mwz", "wo_para", "wo_concat", "vanilla", "open_domain"])
    # parser.add_argument("--train_batch_size", type=int, default=2, help="Batch size for training")
    # parser.add_argument("--dev_batch_size", type=int, default=4, help="Batch size for validation")
    # parser.add_argument("--test_batch_size", type=int, default=4, help="Batch size for test")
    parser.add_argument("--grad_acc_steps", type=int, default=64, help="Accumulate gradients on several steps")
    # parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--num_beams", type=int, default=1, help="Number of beams for beam search during eval")
    parser.add_argument("--test_num_beams", type=int, default=10, help="Number of beams for beam search during test")
    # parser.add_argument("--seed", type=int, default=557, help="Random seed")
    parser.add_argument("--GPU", type=int, default=1, help="how many gpu to use")
    parser.add_argument("--model_name", type=str, default="bart", help="use t5 or bart?")
    parser.add_argument("--fewshot", type=float, default=1.0, help="data ratio for few shot experiment")
    parser.add_argument("--mode", type=str, default="finetune", choices=['finetune', 'pretrain','prefixtune'])
    parser.add_argument("--fix_label", default=True)
    parser.add_argument("--except_domain", type=str, choices=["hotel", "train", "restaurant", "attraction", "taxi"])
    parser.add_argument("--only_domain", type=str, choices=["hotel", "train", "restaurant", "attraction", "taxi"])
    parser.add_argument("--version", type=str, default="2.1" , help="version of multiwoz")
    parser.add_argument("--ignore_or", type=bool, default=True, help="ignore slot with value |. if False, consider only previous one.")
    parser.add_argument("--save_samples", type=int, default=0, help="save # false case samples.")
    # parser.add_argument("--val_check_interval", type=float, default=1.0, help="ratio of train data that should be learned to check validation performance") # conflict with pl.Trainer.add_argparse_args(parser)
    parser.add_argument("--dialogue_filter", type=str, default="min", choices=["max", "min"])
    parser.add_argument("--train_control", type=str, default="none", choices=["selective_rough", "selective_exactly", "previous", "none"])
    parser.add_argument("--load_pretrained", type=str, help="Path to the pretrained CD model")
    parser.add_argument("--debug_code", action='store_true')
    parser.add_argument("--eval_loss_only", action='store_true')
    parser.add_argument("--do_train_only", action='store_true')
    parser.add_argument("--do_test_only", action='store_true')
    parser.add_argument("--resume_from_ckpt", type=str,)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--use_qa_deconverter", action='store_true')
    parser.add_argument("--qa_model_path", type=str)
    parser.add_argument("--balanced_sampling", action="store_true")
    parser.add_argument("--filtered_sampling", action="store_true")

    
    # prefix dependent

    # ['e2e'(E2E dataset), 'cnn_dm'(cnn/daily mail), 'webnlg', 'triples', 'xsum', 'xsum_news', 'xsum_news_sport','multiwoz']
    parser.add_argument('--pf_mode', type=str, default='multiwoz', help='')
    # select tuning mode. ['prefixtune', 'finetune', 'finetune-top', 'bothtune', 'adaptertune']
    # parser.add_argument('--tuning_mode', type=str, default='prefixtune', help='')
    # parser.add_argument('--optim_prefix', type=str, default='yes', help='')
    # select seqlen
    # parser.add_argument('--preseqlen', type=int, default=5, help='')
    # select prefix_tuning mode. ['embedding', 'activation']
    # parser.add_argument('--prefix_mode', type=str, default='activation', help='')
    # select format_mode. ['cat', 'infix', 'peek', 'nopeek']
    # parser.add_argument('--format_mode', type=str, default='cat', help='')

    parser.add_argument('--dir_name', type=str, default=None, help='')
    # parser.add_argument('--notes', type=str, default=None, help='')
    
    # # Maybe, use_lowdata_token indicates whether you reproduce the Figure.5 result in the paper.
    # parser.add_argument('--use_lowdata_token', type=str, default='yes', help='')
    # # Maybe, lowdata_token is the word that initialize the prefix with activations in low-data settings.
    # parser.add_argument('--lowdata_token', type=str, default='summarize', help='')

    # Reparametrization. 
    parser.add_argument('--parametrize_emb', type=str, default='MLP', help='')
    # adapter-related
    parser.add_argument('--adapter_design', type=int, default=1, help='')
    # When tuning_mode is 'finetune-top', select the number of top layers
    parser.add_argument('--top_layers', type=int, default=1, help='')
    # select train or fine-tune.
    # 'do_train' means select one of the methods(prefixtune, adaptertune)
    # parser.add_argument('--do_train', type=str, default='yes', help='')
    # use FP16 bit
    # parser.add_argument('--fp16', type=str, default='no', help='')

    # training parameters.
    # parser.add_argument('--use_dropout', type=str, default='no', help='')
    # parser.add_argument('--seed', type=int, default=101, help='') # old is 42

    parser.add_argument('--bsz', type=int, default=10, help='') # batch size for prefixtuning Model
    parser.add_argument('--use_big', type=str, default='no', help='')
    # parser.add_argument('--epoch', type=int, default=5, help='')
    # parser.add_argument('--max_steps', type=int, default=400, help='')
    parser.add_argument('--eval_steps', type=int, default=50, help='')
    # parser.add_argument('--warmup_steps', type=int, default=100, help='')
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='')
    # parser.add_argument('--learning_rate', type=float, default=5e-05, help='')
    # parser.add_argument('--weight_decay', type=float, default=0.0, help='')
    # parser.add_argument('--dropout', type=float, default=0.0, help='')
    # parser.add_argument('--label_smoothing', type=float, default=0.0, help='')
    # parser.add_argument('--length_pen', type=float, default=1.0, help='') # length penalty
    # parser.add_argument('--mid_dim', type=int, default=512, help='') # mid_dim: the dim of the MLP's middle layer(for reparameterization)
    # parser.add_argument('--use_deep', type=str, default='no', help='')

    parser.add_argument('--prefix_model_path', type=str, default=None, help='')
    parser.add_argument('--finetune_model_path', type=str, default=None, help='')
    parser.add_argument('--submit', type=str, default='no', help='')

    # model specific arguments
    parser = PrefixSummarizationModule.add_model_specific_args(parser, os.getcwd())

    # etc
    parser.add_argument('--log_dir', type=str, default='ds2/logs/', help='')

    args = parser.parse_known_args()[0]


    assert args.optim_prefix in ['yes', 'no']
    if args.optim_prefix == 'yes':
        assert args.preseqlen is not None
    # select tuning methods
    assert args.prefix_mode in ['embedding', 'activation']
    assert args.format_mode in ['cat', 'infix', 'peek', 'nopeek']
    assert args.tuning_mode in ['prefixtune', 'finetune', 'finetune-top', 'bothtune', 'adaptertune']
    if args.prefix_model_path is not None:
        load_prefix_model = True
    else:
        load_prefix_model = False


    assert args.pf_mode == 'multiwoz'
    if args.version == "2.0":
        parser.add_argument('--data_dir', type=str, default="ds2/data_mwoz_2.0/", help='')
    else:
        parser.add_argument('--data_dir', type=str, default="ds2/data_mwoz_2.1/", help='')

    # output_dir
    parser.add_argument('--output_dir', type=str, default="{}{}".format(args.log_dir,args.exp_name), help = '')

    # for ds2 dataloader 
    # add test_batch_size (same as dev_batch_size)
    parser.add_argument('--test_batch_size', type=int, default=args.dev_batch_size, help = '')
    # add lr
    parser.add_argument('--lr', type=float, default=args.learning_rate)


    args = parser.parse_args()
    # args.GPU = [int(gpu) for gpu in args.GPU]
    return args