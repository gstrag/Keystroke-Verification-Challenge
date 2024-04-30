import argparse

configs = argparse.ArgumentParser()


configs.scenario = 'desktop'
configs.experiment_name = 'RNN_model'

configs.base_dir = configs.experiment_name + '/'
configs.data_dir = '../DBs/KVC_data/{}/'.format(configs.scenario)
configs.dev_set = configs.data_dir + '{}_dev_set.npy'.format(configs.scenario)
configs.test_set = configs.data_dir + '{}_test_sessions.npy'.format(configs.scenario)
configs.comparison_file = configs.data_dir + '{}_comparisons.txt'.format(configs.scenario)

configs.model_name = configs.experiment_name + '_' + configs.scenario

configs.log_dir = configs.base_dir + 'training_logs/'
configs.training_log_filename = configs.log_dir + configs.model_name + '_log.txt'
configs.training_log_plot_filename = configs.log_dir + configs.model_name + '_plot.pdf'

configs.model_dir = configs.base_dir + 'models/'
configs.model_filename = configs.model_name + '.pt'
configs.result_dir = configs.base_dir + 'submission_files/'
configs.result_filename = configs.result_dir + '{}_predictions.txt'.format(configs.scenario)

configs.train_val_division = 0.8
configs.batches_per_epoch_train = 16
configs.batches_per_epoch_val = 4
configs.batch_size_train = 512
configs.batch_size_val = 512
configs.sequence_length = 100
configs.num_epochs = 50
