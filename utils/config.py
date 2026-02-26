import os
import argparse

def _upper(s: str) -> str:
    return str(s).strip().upper()

def load_yaml_config(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if cfg is not None else {}

def _maybe_infer_config_from_positional(argv):
    """
    Allow:
      python main.py /path/to/config.yaml
    as a shorthand for:
      python main.py --config /path/to/config.yaml
    """
    if argv is None:
        return argv
    if len(argv) >= 2:
        # argv[0] is program name
        first = argv[1]
        if (not first.startswith("-")) and first.lower().endswith((".yml", ".yaml")):
            # rewrite argv: insert --config
            return [argv[0], "--config", first] + argv[2:]
    return argv

def parse_args_with_config(argv=None):
    # ✅ rewrite argv if user passed positional yaml
    if argv is None:
        import sys
        argv = sys.argv
    argv = _maybe_infer_config_from_positional(argv)

    # stage 1: only --config
    p0 = argparse.ArgumentParser(add_help=False)
    p0.add_argument("--config", type=str, default=None)
    cfg_args, remaining = p0.parse_known_args(argv[1:])

    cfg = {}
    if cfg_args.config is not None:
        if not os.path.isfile(cfg_args.config):
            raise FileNotFoundError(f"--config not found: {cfg_args.config}")
        cfg = load_yaml_config(cfg_args.config)

    def cfg_get(k, default):
        return cfg.get(k, default)

    # stage 2: full args, defaults from yaml
    parser = argparse.ArgumentParser("MORN survival training (CV)")
    parser.add_argument("--config", type=str, default=cfg_args.config)

    parser.add_argument("--dataset", type=str, default=cfg_get("dataset", "KIRP"))
    parser.add_argument("--data_root", type=str, default=cfg_get("data_root", "/data5/zhangye/morn/data/processed"))
    parser.add_argument("--label_root", type=str, default=cfg_get("label_root", "/data5/zhangye/morn/data/label"))

    parser.add_argument("--data_dir", type=str, default=cfg_get("data_dir", None))
    parser.add_argument("--graph_file", type=str, default=cfg_get("graph_file", None))
    parser.add_argument("--label_csv", type=str, default=cfg_get("label_csv", None))

        # ---- task switch (NEW) ----
    parser.add_argument("--task", type=str, default="survival", choices=["survival", "grading"],
                        help="Task type: survival (c-index + NLLSurvLoss) or grading (macro-F1 + CE)")
    parser.add_argument("--grade_label_key", type=str, default="label",
                        help="Node data key for grading labels, e.g., 'grade'/'stage'/'subtype'")
    parser.add_argument("--num_grades", type=int, default=-1,
                        help="Number of grading classes. -1 means infer from labels.")
    parser.add_argument("--use_class_weight", type=int, default=1, choices=[0, 1],
                        help="Use inverse-frequency class weights for CrossEntropy (grading only).")
    parser.add_argument("--label_smoothing", type=float, default=0.0,
                        help="Label smoothing for CrossEntropy (grading only).")

    parser.add_argument("--fold_glob", type=str, default=cfg_get("fold_glob", "splits_*"))
    parser.add_argument("--split_pt_suffix", type=str, default=cfg_get("split_pt_suffix", "_split.pt"))
    parser.add_argument("--meta_json_suffix", type=str, default=cfg_get("meta_json_suffix", "_meta.json"))

    parser.add_argument("--target_ntype", type=str, default=cfg_get("target_ntype", "patient"))
    parser.add_argument("--n_epoch", type=int, default=cfg_get("n_epoch", 100))
    parser.add_argument("--n_hid", type=int, default=cfg_get("n_hid", 256))
    parser.add_argument("--n_layers", type=int, default=cfg_get("n_layers", 2))
    parser.add_argument("--n_heads", type=int, default=cfg_get("n_heads", 4))
    parser.add_argument("--clip", type=float, default=cfg_get("clip", 1.0))
    parser.add_argument("--max_lr", type=float, default=cfg_get("max_lr", 1e-4))
    parser.add_argument("--weight_decay", type=float, default=cfg_get("weight_decay", 0.0))
    parser.add_argument("--eval_every", type=int, default=cfg_get("eval_every", 5))
    parser.add_argument("--alpha_surv", type=float, default=cfg_get("alpha_surv", 0.0))

    parser.add_argument("--edge_weight_key", type=str, default=cfg_get("edge_weight_key", "w"))
    parser.add_argument("--edge_weight_mode", type=str, default=cfg_get("edge_weight_mode", "mul_attn"),
                        choices=["mul_attn", "none"])

    parser.add_argument("--device", type=str, default=cfg_get("device", "cuda:0"))
    parser.add_argument("--seed", type=int, default=cfg_get("seed", 42))

    parser.add_argument("--out_dir", type=str, default=cfg_get("out_dir", None))
    parser.add_argument("--save_model", action="store_true", default=bool(cfg_get("save_model", False)))

    parser.add_argument("--dump_attn", type=int, default=cfg_get("dump_attn", 1), choices=[0, 1])
    parser.add_argument("--dump_attn_at", type=str, default=cfg_get("dump_attn_at", "best"), choices=["best", "final"])
    parser.add_argument("--dump_dirname", type=str, default=cfg_get("dump_dirname", "attn_dump"))
    parser.add_argument("--save_edge_w_in_dump", type=int, default=cfg_get("save_edge_w_in_dump", 1), choices=[0, 1])

    # 注意：这里用 remaining（来自 argv[1:] 的 parse_known_args）
    args = parser.parse_args(remaining)
    args._cfg_dict = cfg
    return args

def resolve_paths(args):
    ds = _upper(args.dataset)

    if args.data_dir is None:
        args.data_dir = os.path.join(args.data_root, f"{ds}_hgt_dataset")
    if args.graph_file is None:
        args.graph_file = f"{ds}_graph.bin"
    if args.label_csv is None:
        args.label_csv = os.path.join(args.label_root, f"{ds}_survival_labels.csv")
    if args.out_dir is None:
        args.out_dir = os.path.join(args.data_dir, "cv_results")

    graph_path = os.path.join(args.data_dir, args.graph_file)
    return ds, graph_path, args.label_csv, args.out_dir
