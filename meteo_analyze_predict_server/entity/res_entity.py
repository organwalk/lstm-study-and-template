from collections import OrderedDict


def model_info(version, cn_des, technology, support, update):
    return OrderedDict([
        ('version', version),
        ('cn_des', cn_des),
        ('technology', technology),
        ('support', support),
        ('update', update)
    ])
