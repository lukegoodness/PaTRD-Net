from data_provider.data_loader import Dataset_Solar, Dataset_Covid, Dataset_Custom, Dataset_Pred, Dataset_Custom_, Dataset_ill, Dataset_ETT_hour, Dataset_ETT_minute
from torch.utils.data import DataLoader
 
data_dict = {
    # ETT benchmarks use their standard fixed monthly splits.
    'ETTh1': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTh2': Dataset_ETT_hour,
    'ETTm2': Dataset_ETT_minute,
    'traffic': Dataset_Custom_,
    'electricity': Dataset_Custom_,
    'exchange': Dataset_Custom_,
    'exchange_rate': Dataset_Custom_,  # 别名，与exchange相同
    'weather': Dataset_Custom_,
    'covid': Dataset_Covid,
    'ECG': Dataset_Custom_,
    'metr': Dataset_Custom_,
    'ill': Dataset_ill,
    'illness': Dataset_ill,  # 别名，与ill相同
    'national_illness': Dataset_ill,  # 别名，与ill相同
    'solar': Dataset_Solar,
    'air': Dataset_Custom_,
}


def data_provider(args, flag):
    # 检查数据集是否存在，如果不存在则尝试别名映射
    if args.data not in data_dict:
        # Normalize common dataset aliases.
        alias_map = {
            'exchange_rate': 'exchange',
            'illness': 'ill',
            'national_illness': 'ill',
            'ILI': 'ill',
            'Weather': 'weather',
            'WEATHER': 'weather',
            'Electricity': 'electricity',
            'ECL': 'electricity',
            'Traffic': 'traffic',
            'Exchange': 'exchange',
            'Solar': 'solar',
        }
        if args.data in alias_map:
            args.data = alias_map[args.data]
        elif args.data.lower() in data_dict:
            # case-insensitive 兜底
            args.data = args.data.lower()
        else:
            raise KeyError(f"数据集 '{args.data}' 不存在。可用的数据集: {list(data_dict.keys())}")
    
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1
    train_only = args.train_only

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    elif flag == 'pred':
        shuffle_flag = False
        drop_last = False
        batch_size = 1
        freq = args.freq
        Data = Dataset_Pred
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq

    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        size=[args.seq_len, args.label_len, args.pred_len],
        features=args.features,
        target=args.target,
        timeenc=timeenc,
        freq=freq,
        train_only=train_only
    )
    print(flag, len(data_set))
    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)
    return data_set, data_loader
