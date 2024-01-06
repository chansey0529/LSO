stage_configs = {
    # For 1-shot setting, best visualization (--single_task) with less LSR steps to prevent overfitting.
    # One shot settings
    'flowers_1': {
        'LAL':{
            'lr': 0.05,
            'kimg': 1,
        },
        'LSR':{
            'lr': 0.0005,
            'kimg': 4,
            'save_magnitude': [1.],
        },
    },
    'animalfaces_1': {
        'LAL':{
            'lr': 0.05,
            'kimg': 1,
        },
        'LSR':{
            'lr': 0.0025,
            'kimg': 10,
            'save_magnitude': [0.9],
        },
    },
    'vggfaces_1': {
        'LAL':{
            'lr': 0.05,
            'kimg': 1,
        },
        'LSR':{
            'lr': 0.0025,
            'kimg': 3,
            'save_magnitude': [0.9],
        },
    },
    # Multi-shot settings
    'flowers': {
        'LAL':{
            'lr': 0.05,
            'kimg': 2,
        },
        'LSR':{
            'lr': 0.0025,
            'kimg': 6,
            'save_magnitude': [0.7],
        },
    },
    'animalfaces': {
        'LAL':{
            'lr': 0.05,
            'kimg': 2,
        },
        'LSR':{
            'lr': 0.0025,
            'kimg': 10,
            'save_magnitude': [0.6],
        },
    },
    'vggfaces': {
        'LAL':{
            'lr': 0.05,
            'kimg': 2,
        },
        'LSR':{
            'lr': 0.0025,
            'kimg': 3,
            'save_magnitude': [0.7],
        },
    },
}

loss_configs = {
    'common':{
        'mgt_lambda': 1.,
        'sreg_lambda': {
            'G': 1.,
            'D': 2.,
        },
        'perc': {
            'l2_lambda': 0.5,
            'lpips_lambda': 1.,
            'id_lambda': 0.5,
        },
    },
    'LAL': {
        'app_lambda': 1.,
    },
    'LSR': {
        'perc_lambda': 1.,
        'adv_lambda': 1.,
    }
}

global_configs = {
    'seen_batch_size': 4,
}