import pandas as pd
import os
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def train_val_split(train_x_seg_path, train_y_seg_path, val_x_seg_path, val_y_seg_path):
    train_df_x = pd.read_csv(train_x_seg_path, encoding='utf-8')
    train_df_y = pd.read_csv(train_y_seg_path, encoding='utf-8')
    print(len(train_df_x))
    print(len(train_df_y))
    # 训练集 验证集划分
    X_train, X_val, y_train, y_val = train_test_split(train_df_x, train_df_y,
                                                      test_size=0.3,  # 0-1之间的数
                                                      shuffle=True,   #默认为true，每次设定random_state为整数（0~42）以保证效果
                                                      random_state=1
                                                      )

    X_train.to_csv(train_x_seg_path, sep='\t', index=None, header=False)
    y_train.to_csv(train_y_seg_path, sep='\t', index=None, header=False)
    X_val.to_csv(val_x_seg_path, sep='\t', index=None, header=False)
    y_val.to_csv(val_y_seg_path, sep='\t', index=None, header=False)


if __name__ == "__main__":
    train_val_split('{}/datasets/train_set.seg_x.txt'.format(BASE_DIR),
                    '{}/datasets/train_set.seg_y.txt'.format(BASE_DIR),
                    '{}/datasets/val_set.seg_x.txt'.format(BASE_DIR),
                    '{}/datasets/val_set.seg_y.txt'.format(BASE_DIR))