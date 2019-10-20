import argparse

from sigtestv.net import replay_data_log


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-file', '-i', type=str, required=True)
    parser.add_argument('--logger-endpoint', '-o', type=str, required=True)
    args = parser.parse_args()
    replay_data_log(args.log_file, args.logger_endpoint)


if __name__ == '__main__':
    main()