import argparse
from wandb_train import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="assist2015")
    parser.add_argument("--model_name", type=str, default="akt_dugp")
    parser.add_argument("--emb_type", type=str, default="qid")
    parser.add_argument("--save_dir", type=str, default="saved_model")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--fold", type=int, default=0)

    # AKT backbone hyperparameters
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--d_ff", type=int, default=256)
    parser.add_argument("--num_attn_heads", type=int, default=8)
    parser.add_argument("--n_blocks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning_rate", type=float, default=1e-5)

    # Formal DUGP-KT parameters
    parser.add_argument("--num_groups", type=int, default=9)          # k + unknown group, k=8 -> 9
    parser.add_argument("--alpha_feat_dim", type=int, default=5)
    parser.add_argument(
        "--dugp_mode",
        type=str,
        default="full",
        choices=[
            "full",
            "fixed_fusion",
            "group_add",
            "group_only",
            "alpha_only",
            "no_distance",
            "no_uncertainty",
            "no_behavior",
            "none",
        ],
    )
    parser.add_argument("--fixed_alpha", type=float, default=0.5)
    parser.add_argument("--alpha_hidden_dim", type=int, default=64)
    parser.add_argument("--alpha_init_bias", type=float, default=2.94443897917)  # sigmoid ~= 0.95
    parser.add_argument("--dugp_layer_norm", type=int, default=0)
    parser.add_argument("--detach_distance", type=int, default=0)
    parser.add_argument("--dugp_residual_scale", type=float, default=0.1)
    parser.add_argument("--learnable_residual_scale", type=int, default=1)

    # Training flags
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--add_uuid", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=200)

    args = parser.parse_args()
    params = vars(args)
    params["dugp_layer_norm"] = bool(params["dugp_layer_norm"])
    params["detach_distance"] = bool(params["detach_distance"])
    params["learnable_residual_scale"] = bool(params["learnable_residual_scale"])
    main(params)
