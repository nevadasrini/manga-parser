from convert_manga109_to_yolo import convert

convert(
    data_root="data/raw/manga109/Manga109_released_2023_12_07",
    output_dir="data/processed/manga109_yolo",
    copy_images=True,
)