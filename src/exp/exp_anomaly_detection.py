import os
import time
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from torch.utils.data import DataLoader
import pdb

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
from utils.anomaly_detection_metrics import bf_search
from core.base_model import BaseAnomalyDetectionModel
import warnings
warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        
    def _build_model(self) -> BaseAnomalyDetectionModel:
        # USAD 모델을 위한 차원 설정
        if self.args.model == 'USAD':
            # PSM 데이터셋의 차원에 맞춰 명시적으로 설정
            self.args.win_size = self.args.seq_len
            self.args.enc_in = self.args.enc_in
            self.args.feature_num = self.args.enc_in
            self.args.window_size = self.args.seq_len
            
            # 디버깅을 위한 로그 출력
            print(f"USAD 모델 차원 설정:")
            print(f"  win_size: {self.args.win_size}")
            print(f"  enc_in: {self.args.enc_in}")
            print(f"  feature_num: {self.args.feature_num}")
            print(f"  window_size: {self.args.window_size}")
            print(f"  seq_len: {self.args.seq_len}")
        
        module = self.model_dict.get(self.args.model)
        if module is None:
            raise ImportError(f"Could not load model module for '{self.args.model}'")
        model = module.Model(self.args).float()
        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.devices)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.model == 'USAD':
            # USAD 모델은 두 개의 optimizer가 필요
            model_optim1 = torch.optim.Adam(self.model.encoder.parameters(), lr=self.args.learning_rate)
            model_optim2 = torch.optim.Adam(self.model.decoder2.parameters(), lr=self.args.learning_rate)
            return (model_optim1, model_optim2)
        else:
            model_optim = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
            return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()  # Use mean reduction for training
        return criterion

    def vali(self, vali_data, vali_loader, criterion) -> Dict[str, float]:
        """
        Perform validation on the validation dataset.

        Args:
            vali_data: Validation dataset
            vali_loader: Validation data loader
            criterion: Loss criterion

        Returns:
            Dictionary containing validation loss
        """
        total_loss = []

        self.model.eval()
        with torch.no_grad():
            for batch in vali_loader:
                # Handle different data loader formats
                if len(batch) == 4:
                    batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                elif len(batch) == 2:
                    batch_x, batch_y = batch
                else:
                    raise ValueError(f"Unexpected batch format with {len(batch)} elements")

                batch_x = batch_x.float().to(self.device)

                # For reconstruction models, target is the input itself
                if self._is_reconstruction_model():
                    batch_y = batch_x
                else:
                    batch_y = batch_y.float().to(self.device)

                # Get model outputs and anomaly scores
                try:
                    outputs, anomaly_scores = self._get_anomaly_scores(batch_x)

                    # Calculate loss
                    if self._is_reconstruction_model():
                        loss = criterion(outputs, batch_x)
                    else:
                        loss = criterion(outputs, batch_y)

                    total_loss.append(loss.item())

                except Exception as e:
                    # Fallback: use simple reconstruction error
                    outputs = self.model(batch_x)
                    loss = criterion(outputs, batch_x)
                    total_loss.append(loss.item())

        avg_loss = np.average(total_loss)
        self.model.train()

        return {'loss': avg_loss}

    def _is_reconstruction_model(self) -> bool:
        """Check if the current model is reconstruction-based."""
        reconstruction_models = [
            'AnomalyTransformer', 'OmniAnomaly', 'USAD', 'DAGMM',
            'LSTM_VAE', 'LSTM_AE', 'VTTPAT', 'VTTSAT'
        ]
        return self.args.model in reconstruction_models
    
    def _get_anomaly_scores(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get anomaly scores using appropriate method based on model type.
        
        Args:
            batch_x: Input batch data
            
        Returns:
            Tuple of (outputs, anomaly_scores)
        """
        if self._is_reconstruction_model():
            return self._reconstruction_based_scoring(batch_x)
        else:
            return self._prediction_based_scoring(batch_x)
    
    def _reconstruction_based_scoring(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get anomaly scores using reconstruction-based method."""
        try:
            # Try to use model's specific anomaly detection method
            if hasattr(self.model, 'detect_anomaly'):
                outputs, scores = self.model.detect_anomaly(batch_x)
            elif hasattr(self.model, 'get_anomaly_score'):
                outputs = self.model(batch_x)
                scores = self.model.get_anomaly_score(batch_x)
            else:
                # Fallback: calculate reconstruction error
                outputs = self.model(batch_x)
                scores = torch.mean((outputs - batch_x) ** 2, dim=-1)
            
            return outputs, scores
            
        except Exception as e:
            # Fallback: simple reconstruction error
            try:
                outputs = self.model(batch_x)
                scores = torch.mean((outputs - batch_x) ** 2, dim=-1)
                return outputs, scores
            except Exception as e2:
                # Return dummy scores
                outputs = batch_x
                scores = torch.zeros(batch_x.shape[0], 1, device=batch_x.device)
                return outputs, scores
    
    def _prediction_based_scoring(self, batch_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get anomaly scores using prediction-based method."""
        try:
            # For prediction-based detection, we need to:
            # 1. Make predictions for the next few steps
            # 2. Calculate prediction variance or error as anomaly score
            
            # Get prediction horizon (use config pred_len or default to 1)
            pred_len = getattr(self.args, 'pred_len', 1)
            
            # Make predictions
            if hasattr(self.model, 'predict_single'):
                predictions = self.model.predict_single(batch_x, pred_len)
            else:
                # Fallback: use model directly
                predictions = self.model(batch_x)
            
            # Calculate prediction error as anomaly score
            if predictions.dim() == 3:  # (batch, pred_len, features)
                # Use prediction variance across time steps as anomaly score
                scores = torch.var(predictions, dim=1, keepdim=True)
            else:
                # Use prediction magnitude as anomaly score
                scores = torch.mean(torch.abs(predictions), dim=-1, keepdim=True)
            
            # For prediction-based, outputs are the predictions
            outputs = predictions
            
            return outputs, scores
            
        except Exception as e:
            # Fallback: use input variance as anomaly score
            try:
                scores = torch.var(batch_x, dim=1, keepdim=True)
                outputs = batch_x
                return outputs, scores
            except Exception as e2:
                # Return dummy scores
                outputs = batch_x
                scores = torch.zeros(batch_x.shape[0], 1, device=batch_x.device)
                return outputs, scores

    def train(self) -> Dict[str, Any]:
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, self.args.des)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
            
        train_history = []
        val_history = []

        # Log detection method being used
        detection_method = "reconstruction-based" if self._is_reconstruction_model() else "prediction-based"
        self.logger.info(f"Using {detection_method} anomaly detection with model: {self.args.model}")

        for epoch in range(self.args.train_epochs):
            train_loss = []
            self.model.train()
            epoch_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                batch_x, batch_y, batch_x_mark, batch_y_mark = batch
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                
                # Model-specific training step (backward와 step이 이미 포함됨)
                loss, score = self.model.train_step(batch_x, batch_y, model_optim, criterion, epoch + 1)
                train_loss.append(loss)

            train_loss = np.average(train_loss)
            vali_result = self.vali(vali_data, vali_loader, criterion)
            vali_loss = vali_result['loss']

            train_metrics = {'loss': train_loss}
            val_metrics = {'loss': vali_loss}
            
            train_history.append(train_metrics)
            val_history.append(val_metrics)

            self.logger.info(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                self.logger.info("Early stopping")
                break

            if isinstance(model_optim, tuple):
                for opt in model_optim:
                    adjust_learning_rate(opt, epoch + 1, self.args)
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        return {
            'train_history': train_history,
            'val_history': val_history,
            'best_model_path': best_model_path,
            'detection_method': detection_method
        }

    def test(self, alpha: float = 0.5, beta: float = 0.5) -> Dict[str, Any]:
        """
        모델 테스트를 수행합니다.

        Args:
            alpha: USAD 모델용 가중치 파라미터
            beta: USAD 모델용 가중치 파라미터

        Returns:
            테스트 결과 딕셔너리
        """
        test_data, test_loader = self._get_data(flag='test')

        if hasattr(self.args, 'test') and self.args.test:
            self.logger.info('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + self.args.des, 'checkpoint.pth')))

        # 테스트 실행
        dist, pred = self._run_test(test_data, test_loader, alpha, beta)
        # 검증 데이터로 임계값 설정 (가능한 경우)
        try:
            vali_data, vali_loader = self._get_data(flag='val')
            vali_result = self.vali(vali_data, vali_loader, self._select_criterion())
            # vali_result is a dict with 'loss' key
            valid_score = None  # Can be used for threshold calculation in future
        except:
            valid_score = None

        # 이상 탐지 메트릭 계산
        # Get labels from dataset object
        labels = None
        if hasattr(test_data, 'label_data'):
            labels = test_data.label_data
        elif hasattr(test_data, 'labels'):
            labels = test_data.labels

        if labels is not None:
            # Convert labels to numpy array and ensure integer type
            if isinstance(labels, (pd.Series, pd.DataFrame)):
                labels = labels.values
            labels = np.array(labels, dtype=np.int32).flatten()

            # Since we predict on windows with stride, we need to align labels with predictions
            # For now, use the first len(dist) labels
            if len(labels) > len(dist):
                # Take corresponding labels based on window indices
                # This assumes test uses stride=seq_len (non-overlapping windows)
                step = len(labels) // len(dist)
                labels = labels[::step][:len(dist)]
            elif len(labels) < len(dist):
                self.logger.warning(f"Labels length ({len(labels)}) < predictions length ({len(dist)})")
                # Pad with zeros or truncate dist
                dist = dist[:len(labels)]
                pred = pred[:len(labels)]

            # PA%K metrics 계산 using bf_search from anomaly_detection_metrics
            history, pa_auc = self._calculate_pa_metrics_with_bf_search(dist, labels)

            # Extract ROC AUC from history if available
            roc_auc = history.get('roc_auc', [0.0])[0] if 'roc_auc' in history else 0.0

            result_dict = {
                'auc': roc_auc,
                'pa_auc': pa_auc,
                'anomaly_score_mean': np.mean(dist),
                'anomaly_score_std': np.std(dist),
                'detection_method': 'model_specific',
                'method_info': {
                    'model_type': self.args.model,
                    'description': 'Uses model-specific anomaly detection logic'
                }
            }

            # Add PA%K individual metrics from history
            for k_idx, k in enumerate(history.get('k_values', [])):
                if k_idx < len(history.get('f1', [])):
                    result_dict[f'f1_{k}'] = history['f1'][k_idx]
                    result_dict[f'precision_{k}'] = history['precision'][k_idx]
                    result_dict[f'recall_{k}'] = history['recall'][k_idx]
                    result_dict[f'roc_auc_{k}'] = history['roc_auc'][k_idx]

            # 성능 지표 결과를 로그로 출력
            self._log_test_results(result_dict, history)

            # Save results
            folder_path = './results/' + self.args.des + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            np.save(folder_path + 'anomaly_scores.npy', dist)
            np.save(folder_path + 'predictions.npy', pred)
            np.save(folder_path + 'labels.npy', labels)
            np.save(folder_path + 'history.npy', history)

            return result_dict
        
        # 기본 결과 반환
        folder_path = './results/' + self.args.des + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path + 'anomaly_scores.npy', dist)
        np.save(folder_path + 'predictions.npy', pred)
        
        basic_result = {
            'anomaly_score_mean': np.mean(dist),
            'anomaly_score_std': np.std(dist),
            'detection_method': 'model_specific',
            'method_info': {
                'model_type': self.args.model,
                'description': 'Uses model-specific anomaly detection logic'
            }
        }
        
        # 기본 결과도 로그로 출력
        self._log_basic_test_results(basic_result)
        
        return basic_result

    def _run_test(self, test_data: Any, test_loader: Any, alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
        """테스트를 실행하여 거리와 예측값을 반환합니다.

        Args:
            test_data: Test dataset object
            test_loader: Test data loader
            alpha: USAD alpha parameter
            beta: USAD beta parameter

        Returns:
            Tuple of (anomaly_scores, predictions)
        """
        dist = []
        pred = []
        criterion = self._select_criterion()

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                # 모델별 테스트 로직 실행
                batch_pred, batch_dist = self.model.test_step(
                    batch_x, batch_y, criterion
                )

                # Convert to numpy if tensors
                if isinstance(batch_pred, torch.Tensor):
                    batch_pred = batch_pred.cpu().numpy()
                if isinstance(batch_dist, torch.Tensor):
                    batch_dist = batch_dist.cpu().numpy()

                pred.append(batch_pred)
                dist.append(batch_dist)

        # 결과 결합
        dist = np.concatenate(dist).flatten()
        pred = np.concatenate(pred)

        return dist, pred

    def _calculate_pa_metrics_with_bf_search(self, dist: np.ndarray, labels: np.ndarray,
                                              K_VALUES: List[int] = None) -> Tuple[Dict[str, Any], float]:
        """
        PA%K 메트릭을 계산합니다.

        Args:
            dist: 이상 점수 배열
            labels: 실제 라벨 배열
            K_VALUES: K 값 리스트

        Returns:
            Tuple of (history, pa_auc)
        """
        if K_VALUES is None:
            K_VALUES = [0, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

        unique_classes = np.unique(labels)
        if len(unique_classes) == 1:
            self.logger.warning(f"Only one class present in labels: {unique_classes[0]}")
            return {'k_values': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': [], 'threshold': []}, 0.0

        history = {
            'k_values': [],
            'f1': [],
            'precision': [],
            'recall': [],
            'roc_auc': [],
            'threshold': []
        }

        f1_values = []

        start_threshold = np.percentile(dist, 90)
        end_threshold = np.percentile(dist, 99)
        self.logger.info(f'Threshold start: {start_threshold:.4f} end: {end_threshold:.4f}')

        for k in K_VALUES:
            try:
                # bf_search를 사용하여 최적의 임계값과 메트릭 계산
                [f1, precision, recall, _, _, _, _, roc_auc, _, _], threshold = bf_search(
                    dist, labels,
                    start=np.percentile(dist, 50),
                    end=np.percentile(dist, 99),
                    step_num=1000,
                    K=k,
                    verbose=False
                )

                f1_values.append(f1)
                history['k_values'].append(k)
                history['f1'].append(f1)
                history['precision'].append(precision)
                history['recall'].append(recall)
                history['roc_auc'].append(roc_auc)
                history['threshold'].append(threshold)

                self.logger.info(f"K: {k} precision: {precision:.4f} recall: {recall:.4f} f1: {f1:.4f} AUROC: {roc_auc:.4f}")

            except ValueError as e:
                if "Only one class present in y_true" in str(e):
                    self.logger.warning(f"ROC AUC cannot be calculated for K={k}. Using default values.")
                    f1_values.append(0.0)
                    history['k_values'].append(k)
                    history['f1'].append(0.0)
                    history['precision'].append(0.0)
                    history['recall'].append(0.0)
                    history['roc_auc'].append(0.0)
                    history['threshold'].append(np.percentile(dist, 90))
                else:
                    raise e

        # PA%K AUC 계산
        pa_auc = 0
        for i in range(len(K_VALUES) - 1):
            pa_auc += 0.5 * (f1_values[i] + f1_values[i + 1]) * (int(K_VALUES[i + 1]) - int(K_VALUES[i]))
        pa_auc /= 100

        self.logger.info(f'PA%K AUC: {pa_auc:.4f}')

        return history, pa_auc

    def _log_test_results(self, result_dict: Dict[str, Any], history: Dict[str, Any]) -> None:
        """테스트 결과를 로그로 출력합니다."""
        self.logger.info("=" * 80)
        self.logger.info("🚀 ANOMALY DETECTION TEST RESULTS")
        self.logger.info("=" * 80)

        # 기본 정보
        self.logger.info(f"📊 Model: {self.args.model}")
        self.logger.info(f"📁 Dataset: {self.args.data}")
        self.logger.info(f"🔍 Detection Method: {result_dict['detection_method']}")
        self.logger.info(f"📈 Description: {result_dict['method_info']['description']}")

        # 기본 메트릭
        self.logger.info("-" * 50)
        self.logger.info("📊 BASIC METRICS")
        self.logger.info("-" * 50)
        self.logger.info(f"🎯 ROC-AUC: {result_dict['auc']:.6f}")
        self.logger.info(f"📈 PA%K AUC: {result_dict['pa_auc']:.6f}")
        self.logger.info(f"📊 Anomaly Score Mean: {result_dict['anomaly_score_mean']:.6f}")
        self.logger.info(f"📊 Anomaly Score Std: {result_dict['anomaly_score_std']:.6f}")

        # PA%K 개별 메트릭
        self.logger.info("-" * 50)
        self.logger.info("📊 PA%K INDIVIDUAL METRICS")
        self.logger.info("-" * 50)

        # K 값별로 그룹화하여 출력
        k_values = history.get('k_values', [])

        for k_idx, k in enumerate(k_values):
            if k_idx < len(history.get('f1', [])):
                f1 = history['f1'][k_idx]
                precision = history['precision'][k_idx]
                recall = history['recall'][k_idx]
                auc = history['roc_auc'][k_idx]

                self.logger.info(f"🎯 PA%{k:02d}: F1={f1:.6f}, P={precision:.6f}, R={recall:.6f}, AUC={auc:.6f}")

        # 임계값 정보
        if 'threshold' in history and history['threshold']:
            self.logger.info("-" * 50)
            self.logger.info("🔍 THRESHOLD INFORMATION")
            self.logger.info("-" * 50)
            self.logger.info(f"📊 First Threshold: {history['threshold'][0]:.6f}")

        # 요약
        self.logger.info("-" * 50)
        self.logger.info("📋 SUMMARY")
        self.logger.info("-" * 50)
        self.logger.info(f"✅ Test completed successfully for {self.args.model}")
        self.logger.info(f"📁 Results saved to: ./results/{self.args.des}/")
        if history.get('f1', []):
            self.logger.info(f"🎯 Best PA%K F1: {max(history['f1']):.6f}")
        self.logger.info("=" * 80)
    
    def predict_single(self, input_data: torch.Tensor, num_steps: int = 1) -> torch.Tensor:
        """
        Make prediction for single input (mainly for prediction-based models).
        
        Args:
            input_data: Input tensor
            num_steps: Number of steps to predict ahead
            
        Returns:
            Predictions tensor
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'predict_single'):
                predictions = self.model.predict_single(input_data, num_steps)
            else:
                # Fallback: use model directly
                predictions = self.model(input_data)
        
        return predictions
    
    def detect_anomaly(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Detect anomalies in input data.

        Args:
            input_data: Input tensor

        Returns:
            Anomaly scores tensor
        """
        self.model.eval()
        with torch.no_grad():
            outputs, scores = self._get_anomaly_scores(input_data)

        return scores