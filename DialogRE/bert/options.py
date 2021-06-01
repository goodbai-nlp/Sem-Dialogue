# coding:utf-8
import argparse

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bert_config_file",
        default=None,
        type=str,
        required=True,
        help="The config json file corresponding to the pre-trained BERT model. \n"
        "This specifies the model architecture.",
    )
    parser.add_argument(
        "--task_name", default=None, type=str, required=True, help="The name of the task to train."
    )
    parser.add_argument(
        "--vocab_file",
        default=None,
        type=str,
        required=True,
        help="The vocabulary file that the BERT model was trained on.",
    )
    parser.add_argument(
        "--architecture", default="Hier", type=str, required=True, help="The name of the model to train."
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model checkpoints will be written.",
    )
    parser.add_argument(
        "--init_checkpoint",
        default=None,
        type=str,
        help="Initial checkpoint (usually from a pre-trained BERT model).",
    )
    parser.add_argument(
        "--do_lower_case",
        default=False,
        action="store_true",
        help="Whether to lower case the input text. True for uncased models, False for cased models.",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after WordPiece tokenization. \n"
        "Sequences longer than this will be truncated, and sequences shorter \n"
        "than this will be padded.",
    )
    parser.add_argument(
        "--do_train", default=False, action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", default=False, action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_evalc", default=False, action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--train_batch_size", default=32, type=int, help="Total batch size for training."
    )
    parser.add_argument(
        "--eval_batch_size", default=32, type=int, help="Total batch size for eval."
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--warmup_proportion",
        default=0.1,
        type=float,
        help="Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10%% of training.",
    )
    parser.add_argument(
        "--save_checkpoints_steps",
        default=1000,
        type=int,
        help="How often to save the model checkpoint.",
    )
    parser.add_argument(
        "--no_cuda",
        default=False,
        action="store_true",
        help="Whether not to use CUDA when available",
    )
    parser.add_argument(
        "--local_rank", type=int, default=-1, help="local_rank for distributed training on gpus"
    )
    parser.add_argument("--seed", type=int, default=666, help="random seed for initialization")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumualte before performing a backward/update pass.",
    )
    parser.add_argument(
        "--optimize_on_cpu",
        default=False,
        action="store_true",
        help="Whether to perform optimization and keep the optimizer averages on CPU",
    )
    parser.add_argument(
        "--fp16",
        default=False,
        action="store_true",
        help="Whether to use 16-bit float precision instead of 32-bit",
    )
    parser.add_argument(
        "--loss_scale",
        type=float,
        default=128,
        help="Loss scaling, positive power of 2 values can improve fp16 convergence.",
    )
    parser.add_argument(
        "--resume", default=False, action="store_true", help="Whether to resume the training."
    )
    parser.add_argument(
        "--f1eval",
        default=True,
        action="store_true",
        help="Whether to use f1 for dev evaluation during training.",
    )
    parser.add_argument(
        "--model_type",
        default="std",
        type=str,
        help="Whether to use f1 for dev evaluation during training.",
    )
    parser.add_argument(
        "--entity_drop", default=0.1, type=float, help="entity dropout during training.",
    )
    parser.add_argument(
        "--save_data", default='workplace/data-bin', type=str, help="entity dropout during training.",
    )
    parser.add_argument("--d_concept", type=int, default=768, help="dimension of graph layer")
    parser.add_argument("--d_relation", type=int, default=64, help="dimension of relation embedding")
    parser.add_argument("--g_num_layer", type=int, default=2, help="number of graph layers")
    parser.add_argument(
        "--g_pe",
        default=False,
        action="store_true",
        help="Whether to use position embedding for graph encoder.",
    )
    parser.add_argument("--g_d_ff", type=int, default=1024, help="dimension of graph ff layer") 
    return parser