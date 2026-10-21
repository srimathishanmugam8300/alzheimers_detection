import splitfolders

splitfolders.ratio(
    r'D:\my_projects\alzheimers_detection\data\raw\dataset\Data',
    output=r'D:\my_projects\alzheimers_detection\data\processed',
    seed=42,
    ratio=(0.64, 0.16, 0.20)
)
