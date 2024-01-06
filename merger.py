import os
import re
import argparse


def print_path(path):
    for _drs, dds, _dfs in os.walk(path):
        dds.sort()
        for d in dds:
            print(' '*3, d, len(os.listdir(os.path.join(path, d))))
        break

if __name__ == '__main__':
    
    '''
    Merging the generated images of separated runs (different categories) for evaluation.
    Separate runs can be executed by specifying "--classes <start_idx>-<end_idx>" when running train_unseen.py.

    Template:

    python merger.py \\
        --path <output_dir> \\
        --idx --idx <runidx_1>,...,<runidx_n>

    Examples:

    Merge "00000-xxx" and "00001-xxx" under "output" folder.

        python merger.py \\
            --path ./output \\
            --idx 0,1

    Results are merged to another new run directory with a new number under the same parent folder.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, help='Directory of the parent folder of output runs.', required=True)
    parser.add_argument('--idx', type=str, help='Indices of separated runs, seprated by comma.', required=True)
    args = parser.parse_args()

    target_list = args.idx.split(',')
    postfix = '_'.join(target_list)
    target_list = [int(t) for t in target_list]
    prev_run_dirs = []
    if os.path.isdir(args.path):
        prev_run_dirs = [x for x in os.listdir(args.path) if os.path.isdir(os.path.join(args.path, x))]
    prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
    prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
    cur_run_id = max(prev_run_ids, default=-1) + 1
    run_dir = os.path.join(args.path, f'{cur_run_id:05d}-merger_{postfix}')
    assert not os.path.exists(run_dir)

    os.makedirs(run_dir)
    for roots, dirs, files in os.walk(args.path):
        dirs = [dir for dir in dirs if int(dir.split('-')[0]) in target_list]
        dirs.sort(key=lambda x: int(x.split('-')[0]))
        print("All dirs---------------------------")
        for dir in dirs:
            print(dir)
            print_path(os.path.join(args.path, dir))
        print('-----------------------------------')
        for dir in dirs:
            os.system(f'cp -r {os.path.join(roots, dir)}/* {run_dir}')
        print(run_dir)
        print_path(run_dir)
        break