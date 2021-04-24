import abc
import torch
from train.model.models import FM, DeepFM
from train.metrics import logloss, pr_auc
from torch import optim


class Trainer(abc.ABC):
    """モデルの学習を行う抽象クラス"""

    def __init__(self, loss_fn, optimizer, device, hypara_dict, save_dir='/workspace/tmp_storage'):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.save_dir = save_dir
        self.hypara_dict = hypara_dict

    def fit(self, data_loader, valid_X, valid_y, epochs, model_params_file='tmp_params.pth'):
        """モデルの学習を実行する"""
        model = self._build_model(self.hypara_dict)
        model = model.to(self.device)
        if self.optimizer == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=0.001)
        if self.optimizer == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        if self.optimizer == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=0.001)

        for epoch in range(epochs):
            model.train()
            max_auc = 0.0
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()  # 勾配初期化
                logits = model(inputs)
                loss = self.loss_fn(logits, labels)
                loss.backward()  # back propagetion
                optimizer.step()  # パラメータ更新
            _, valid_auc = self._validation_model(model, valid_X, valid_y)
            if valid_auc > max_auc:
                self._save_model_params(model, model_params_file)
                max_auc = valid_auc

    @abc.abstractmethod
    def _build_model(self, hypara_dict):
        """モデルを構築する"""
        pass

    def _validation_model(self, model, valid_X, valid_y):
        """"validation dataを評価する"""
        model.eval()
        inputs = valid_X.to(self.device)
        with torch.no_grad():
            pred_probs = torch.sigmoid(model(inputs))
        pred_probs = pred_probs.to('cpu').detach().numpy()
        loss = logloss(valid_y, pred_probs)
        auc = pr_auc(valid_y, pred_probs)
        return loss, auc
    
    def _save_model_params(self, model, model_params_file):
        """モデルのパラメータ保存"""
        torch.save(model.state_dict(), f'{self.save_dir}/{model_params_file}')
    
    def predict(self, eval_X, model_params_file='tmp_params.pth'):
        """テストデータを推論"""
        model = self._build_model(self.hypara_dict)
        model.load_state_dict(torch.load(f'{self.save_dir}/{model_params_file}'))
        model.eval()
        inputs = eval_X.to(self.device)
        with torch.no_grad():
            pred_probs = torch.sigmoid(model(inputs))
        return pred_probs


class FMTrainer(Trainer):
    """Factorization Machineの学習を行うクラス"""
    def _build_model(self, hypara_dict):
        '''モデルを構築する'''
        embed_rank = hypara_dict['embed_rank']
        field_dims = hypara_dict['field_dims']
        model = FM(field_dims, embed_rank)
        return model


class DeepFMTrainer(Trainer):
    """DeepFMの学習を行うクラス"""
    def _build_model(self, hypara_dict):
        '''モデルを構築する'''
        embed_rank = hypara_dict['embed_rank']
        field_dims = hypara_dict['field_dims']
        mlp_dims = hypara_dict['mlp_dims']
        drop_out = hypara_dict['drop_out']
        model = DeepFM(field_dims, embed_rank, mlp_dims, drop_out)
        return model