from pandas import read_parquet
import argparse
from os.path import join
from glob import glob


def parquet_to_dataframe(file, save_to_csv):
    df = read_parquet(file, engine='pyarrow')

    if save_to_csv:
        df.to_csv(file.replace('parquet', 'csv'), index=False)
    return df


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Parquet to csv')
    parser.add_argument('--task_folder', help='Folder in results package containing parquet files', required=True)
    parser.add_argument('--files_pattern', help='Pattern for files to read', required=True)
    args = parser.parse_args()

    task_folder = args.task_folder
    file_pattern = args.files_pattern

    file_names = glob(join(task_folder, file_pattern))

    for filename in file_names:
        df = parquet_to_dataframe(filename, True)
        print(df)

    # Execution example
    # python parquet_reader.py --task_folder=task1 --files_pattern=lda_topics_k_4_*