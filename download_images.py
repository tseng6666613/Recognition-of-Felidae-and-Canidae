import os
from icrawler.builtin import BingImageCrawler

# 定義各類別與其關鍵字
categories = {
    'none': ['car', 'mountain', 'human', 'bird', 'cow', 'horse'],  # 其他類別當成 none
    'leopard': ['leopard', 'wild leopard'],
    'tiger': ['tiger', 'bengal tiger'],
    'lion': ['lion', 'wild lion'],
    'dog': ['dog', 'puppy'],
    'wolf': ['wolf', 'gray wolf'],
    'fox': ['fox', 'red fox'],
    'cat': ['cat', 'kitten']
}

# 每組關鍵字最多抓幾張圖
max_per_keyword = 120

# 圖片儲存路徑
base_dir = 'dataset'

for category, keywords in categories.items():
    category_dir = os.path.join(base_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    for keyword in keywords:
        crawler = BingImageCrawler(storage={'root_dir': category_dir})
        crawler.crawl(keyword=keyword, max_num=max_per_keyword)
