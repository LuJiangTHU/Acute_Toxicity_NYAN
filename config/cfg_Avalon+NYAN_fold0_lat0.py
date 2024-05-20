config = {

    'train_fold_file': 'train_fold_0.txt',
    'test_fold_file': 'test_fold_0.txt',

    'feature_name': 'Avalon + NYAN0',


    'max_num_epochs': 100,

    'data_train_opt': {
        'batch_size': 32,
        'epoch_size': 100
    },

    'data_test_opt': {
        'batch_size': 24025,
        'epoch_size': 1
    },

    'learners': {

        'DNN': {
            'def_file': './models/DNN.py',
            'pretrained': None,

            'opt': {
                'in_features': 1088,
                'out_features': [1500,500,100],
                'num_layers': 3,
                'Dropout_p': 0.1

            },

            'optim_params': {
                'optim_type': 'sgd',
                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'nesterov': True,
                'LUT_lr': [(20, 0.01), (40, 0.006), (50, 0.0012), (60, 0.00024),
                           (70, 0.00012), (80, 0.00001), (100, 0.000001)]
            }

        },


        'Regressor': {
            'def_file': 'models/Regressor.py',
            'pretrained': None,

            'opt': {
                'in_features': 100,
                'out_tasks': 59
            },

            'optim_params': {
                'optim_type': 'sgd',
                'lr': 0.1,
                'momentum': 0.9,
                'weight_decay': 5e-4,
                'nesterov': True,
                'LUT_lr': [(20, 0.01), (40, 0.006), (50, 0.0012), (60, 0.00024),
                           (70, 0.00012), (80, 0.00001), (100, 0.000001)]
            }
        }
    }
}





