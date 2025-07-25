DEFAULT_CONFIG = {
    'scenarios': {
        'perfect_1_1': 30,
        'installments_1_n': 30,
        'group_payment_n_1': 30,
        'standalone_invoices': 5,
        'standalone_payments': 5
    },
    'num_companies': 20,
    'recurrency_patterns': {
        'recurring_clients': 0.3,
        'similar_services': 0.4,
        'monthly_services': 0.2,
        'project_based': 0.3
    },
    'quality_distribution': {
        'perfect': 0.8,
        'fuzzy': 0.1,
        'noisy': 0.1
    },
    'timing_distribution': {
        'standard': 0.6,
        'delayed': 0.2,
        'early': 0.1,
        'same_day': 0.1
    },
    'amount_distribution': {
        'exact': 0.8,
        'partial': 0.05,
        'excess': 0.0,
        'discount': 0.1,
        'penalty': 0.04
    },
    'batch_size': 10
}

