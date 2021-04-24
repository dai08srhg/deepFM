import pandas as pd
from sklearn.model_selection import train_test_split
from train.dataset import FMDataset
from train.trainer import DeepFMTrainer
import torch
from torch import nn, optim

def main():
    df = pd.read_csv('data_path')
    df_train, df_eval = train_test_split(df, test_size=0.5, train_size=0.5, random_state=48, shuffle=True)
    df_valid, df_test = train_test_split(df_eval, test_size=0.5, train_size=0.5, random_state=48, shuffle=True)
    
    input_columns = list(df_train.columns)
    label_column = 'true_label'
    dataset = FMDataset(df_train, df_valid, df_test, input_columns, label_column)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = 'RMSprop'
    device = 'cpu'
    field_dims = dataset.field_dims
    hypara_dict = {'embed_rank': 10, 'field_dims': field_dims, 'mlp_dims': (64, 32), 'drop_out': 0.25}

    trainer = DeepFMTrainer(loss_fn, optimizer, device, hypara_dict)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size = 64, shuffle = True, num_workers = 2)
    valid_X, valid_y = dataset.get_valid()

    trainer.fit(train_loader, valid_X, valid_y, epochs=100)

    test_X, test_y = dataset.get_test()
    pred_probs = trainer.predict(test_X)


if __name__ == '__main__':
    main()