from pathlib import Path
import argparse
import re
import os
import subprocess
import tempfile


def set_jiant_setting(config_str, name, value):
    m = re.search(rf'{name}\s*=\s*', config_str)
    if not m:
        return f'{config_str}\n{name} = {value}'
    return re.sub(rf'({name}\s*=\s*).+', rf'\g<1>{value}', config_str)


def set_jiant_settings(config_str, **kwargs):
    for key, value in kwargs.items():
        config_str = set_jiant_setting(config_str, key, value)
    return config_str


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--project-prefix', '-pp', type=Path, required=True)
    parser.add_argument('--data-dir', '-d', type=Path, required=True)
    parser.add_argument('--word-embeddings-file', '-emb', type=Path, required=True)
    parser.add_argument('--config-file', '-c', type=Path, required=True)
    parser.add_argument('--run-name', type=str, required=True)
    parser.add_argument('--base-command', '-bc', type=str, default='python -m jiant')
    parser.add_argument('--cuda-no', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--transfer-nonstatic', action='store_true')
    parser.add_argument('--task-name', type=str, default='sts-b', choices=['sts-b', 'sst'])
    args = parser.parse_args()

    env = os.environ.copy()
    env.update(dict(JIANT_PROJECT_PREFIX=args.project_prefix,
                    JIANT_DATA_DIR=args.data_dir,
                    WORD_EMBS_FILE=args.word_embeddings_file))

    with tempfile.NamedTemporaryFile(dir=os.path.dirname(args.config_file)) as f, open(args.config_file) as f2:
        cfg = f2.read()
        cfg = set_jiant_settings(cfg,
                                 random_seed=args.seed,
                                 cuda=args.cuda_no,
                                 lr=args.lr,
                                 run_name=args.run_name,
                                 target_tasks=args.task_name,
                                 patience=args.patience,
                                 transfer_paradigm='finetune' if args.transfer_nonstatic else 'frozen')
        f.write(cfg.encode())
        f.flush()
        command = f'{args.base_command} --config_file {f.name}'
        p = subprocess.Popen(command, shell=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
        for line in iter(p.stdout.readline, b''):
            print(line.decode(), end='')


if __name__ == '__main__':
    main()