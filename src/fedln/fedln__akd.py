# Nousphera maintenance: dependency-free help and bounded runtime setup.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH']='true'
os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from time import sleep
import glob
import argparse
from utils.runtime import temporary_cache_directory

from utils.optimizers import add_paper_arguments, resolve_optimizer, make_optimizer

parser = argparse.ArgumentParser()
parser.add_argument('--num_clients',			type=int,	default=3,					required=False)
parser.add_argument('--num_rounds',				type=int,	default=10,					required=False)
parser.add_argument('--participation_rate',		type=float,	default=1.0,				required=False)
parser.add_argument('--batch_size',				type=int,	default=128,				required=False)
parser.add_argument('--train_epochs',			type=int,	default=1,					required=False)
parser.add_argument('--lr',						type=float,	default=1e-3,				required=False)
parser.add_argument('--knn_neighbors',			type=int,	default=10,					required=False)
parser.add_argument('--noisy_frac',				type=float,	default=0.8,				required=False)
parser.add_argument('--noise_level',			type=float,	default=0.4,				required=False)
parser.add_argument('--noise_sparsity',			type=float,	default=0.7,				required=False)
parser.add_argument('--distil_round',			type=int,	default=1,					required=False)
parser.add_argument('--embeddings_dir',			type=str,	default='./data/features',	required=False)
parser.add_argument('--embeddings_dims',		type=int,	default=512,				required=False)
parser.add_argument('--dataset_name',			type=str,	default='eurosat',			required=False)
parser.add_argument('--model_name',				type=str, 	default='resnet20',			required=False)
parser.add_argument('--temp_dir',				type=str, 	default='./tmp',			required=False, help='Parent directory for a private disposable simulation cache')
parser.add_argument('--seed',					type=int, 	default=42,					required=False)
parser.add_argument('--akd_teacher', choices=['embeddings', 'global_logits'], default='embeddings')
parser.add_argument('--kd_beta', type=float, default=10.0)
parser.add_argument('--temperature', type=float, default=2.0)
parser.add_argument('--noise_threshold', type=float, default=0.15)
add_paper_arguments(parser)
args = resolve_optimizer(parser.parse_args())
if args.kd_beta < 0 or args.temperature <= 0 or not 0 <= args.noise_threshold <= 1:
	parser.error('Invalid beta, temperature, or noise threshold')

import flwr as fl
from utils.features import load_feature_rows, source_digest
from utils.gpu_utils import get_available_gpu as grab_gpu


def load_available_datasets(train=True):
	import data
	return data.registry(train,args)


def load_available_models():
	import models
	return {
		'resnet20': models.load_resnet20_model,
		'cnn': models.load_cnn_model,
	}

def create_client(cid):
	if os.environ.get('FEDLN_DEVICE') != 'cpu': sleep(int(cid)*0.75)
	os.environ['CUDA_DEVICE_ORDER']="PCI_BUS_ID"
	os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
	sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
	from utils.flwr_client import _Client as Client
	from utils.knn_relabel import estimate_noise_with_pretrained_knn
	from utils.distiller import Distiller, GlobalLogitDistiller
	import numpy as np
	import tensorflow as tf

	#############################################
	# Override Client to use Noise-Aware FedAvg #
	#############################################
	class AKDClient(Client):
		"""Keep the embedding projection local; transmit classifier weights only."""
		def __init__(self, dataset, distil_round, num_neighbors, temp_dir, *pos, **kwargs):
			super().__init__(*pos, **kwargs)
			self.distil_round = distil_round
			self.dataset = dataset
			self.num_neighbors = num_neighbors
			self.data_loader = kwargs['data_loader']
			self.num_clients = int(kwargs['num_clients'])
			self.temp_dir = temp_dir
			self.loader_kwargs = {k: kwargs[k] for k in
				('batch_size', 'seed', 'noisy_clients_frac', 'noise_lvl', 'noise_sparsity')}
			self.noise_path = f'{temp_dir}/noise_{self.cid}.npy'
			self.projection_path = f'{temp_dir}/projection_{self.cid}.npz'

		@property
		def noisy(self):
			return os.path.isfile(self.noise_path) and float(np.load(self.noise_path)[0]) > 0

		def set_parameters(self, parameters, config):
			with_embeddings = self.noisy and args.akd_teacher == 'embeddings'
			if not hasattr(self, 'student') or self._with_embeddings != with_embeddings:
				self._with_embeddings = with_embeddings
				self.student = self.model_loader(input_shape=self.input_shape[1:], num_classes=self.num_classes,
					embeddings_dim=int(args.embeddings_dims) if with_embeddings else None)
				self.classifier = tf.keras.Model(self.student.inputs, self.student.outputs[0]) if with_embeddings else self.student
				if with_embeddings:
					self.projection = self.student.outputs[1]._keras_history.layer
					if os.path.isfile(self.projection_path):
						with np.load(self.projection_path) as saved:
							self.projection.set_weights([saved[k] for k in saved.files])
			if parameters is not None: self.classifier.set_weights(parameters)

		def get_parameters(self, config=None):
			return self.classifier.get_weights()

		def fit(self, parameters, config):
			_data, _, _, labels, idxs = self.data_loader(shard_id=int(self.cid), num_shards=self.num_clients,
				shuffle=False, **self.loader_kwargs)
			idxs = np.asarray(idxs).reshape(-1)
			features = None
			if int(config['round']) >= self.distil_round and not os.path.isfile(self.noise_path):
				features = load_feature_rows(args.embeddings_dir,self.dataset,idxs, source_digest(self.dataset,args.data_root) if self.dataset!='synthetic' else None)
				estimated_noise = estimate_noise_with_pretrained_knn(labels, features, self.num_classes,
					num_neighbors=self.num_neighbors, noise_theshold=args.noise_threshold)
				os.makedirs(self.temp_dir, exist_ok=True)
				np.save(self.noise_path, [estimated_noise])
			self.set_parameters(parameters, config)
			self.model = self.student
			self.data = _data
			if self._with_embeddings:
				if features is None: features = load_feature_rows(args.embeddings_dir,self.dataset,idxs, source_digest(self.dataset,args.data_root) if self.dataset!='synthetic' else None)
				if features.shape[1] != int(args.embeddings_dims):
					raise ValueError('Embedding width does not match --embeddings_dims')
				self.model = Distiller(self.student, features, idxs)
				self.data = tf.data.Dataset.zip((_data.unbatch(), tf.data.Dataset.from_tensor_slices(idxs)))
				self.data = self.data.batch(self.loader_kwargs['batch_size']).prefetch(tf.data.AUTOTUNE)
			elif self.noisy:
				teacher = tf.keras.models.clone_model(self.classifier)
				teacher.set_weights(self.classifier.get_weights())
				self.model = GlobalLogitDistiller(self.student, teacher, args.kd_beta, args.temperature)
			compile_options = {'alpha':args.kd_beta} if self._with_embeddings else {}
			self.model.compile(optimizer=make_optimizer(config),
				loss=tf.keras.losses.CategoricalCrossentropy(name='loss', from_logits=True),
				metrics=[tf.keras.metrics.CategoricalAccuracy(name='accuracy')], **compile_options)
			h = self.model.fit(self.data, epochs=config['epochs'], verbose=0)
			if self._with_embeddings: np.savez(self.projection_path, *self.projection.get_weights())
			metrics = {'accuracy':float(h.history['accuracy'][-1]), 'loss':float(h.history['loss'][-1]),
				'distillation_active': bool(self.noisy), 'teacher':args.akd_teacher, 'kd_beta':args.kd_beta, 'temperature':args.temperature, 'cid': self.cid}
			return self.get_parameters(), self.num_samples, metrics

	#############################################

	load_model = load_available_models()[args.model_name]
	load_train_data = load_available_datasets()[args.dataset_name]
	kwargs = {'batch_size':int(args.batch_size), 'seed':int(args.seed), 'noisy_clients_frac':float(args.noisy_frac),
			'noise_lvl':float(args.noise_level), 'noise_sparsity':float(args.noise_sparsity)}
	return AKDClient(dataset=args.dataset_name, distil_round=int(args.distil_round), num_neighbors=int(args.knn_neighbors),
            temp_dir=args.temp_dir, cid=cid, num_clients=int(args.num_clients), model_loader=load_model, data_loader=load_train_data,
            shuffle=False, **kwargs)

def create_server():
	os.environ['CUDA_VISIBLE_DEVICES'] = grab_gpu()
	from utils.flwr_server import _Server as Server
	load_model = load_available_models()[args.model_name]
	load_test_data = load_available_datasets(train=False)[args.dataset_name]
	kwargs = {'lr':float(args.lr), 'train_epochs':int(args.train_epochs),
              'optimizer':args.optimizer,'momentum':args.momentum}
	return Server(num_rounds=int(args.num_rounds), num_clients=int(args.num_clients), participation=float(args.participation_rate),
            model_loader=load_model, data_loader=load_test_data, **kwargs)

def run_simulation():
	# Only a newly allocated child directory belongs to this simulation.
	cache_parent = args.temp_dir
	with temporary_cache_directory(cache_parent) as cache_dir:
		args.temp_dir = cache_dir
		try:
			# Create server
			server = create_server()
			# Start simulation
			history = fl.simulation.start_simulation(client_fn=create_client, server=server, num_clients=int(args.num_clients),
				ray_init_args= {"ignore_reinit_error": True, "num_cpus": int(args.num_clients),},
				config=fl.server.ServerConfig(num_rounds=int(args.num_rounds), round_timeout=None),)
			return history
		finally:
			args.temp_dir = cache_parent


if __name__ == "__main__":
	print(run_simulation())
