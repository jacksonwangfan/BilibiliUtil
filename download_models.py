import os
import requests
import tarfile
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def extract_tar(filename):
    """Extract tar file"""
    with tarfile.open(filename, 'r') as tar:
        tar.extractall()
    os.remove(filename)

def main():
    # 模型文件URLs
    models = {
        'det': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar',
        'rec': 'https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar',
        'cls': 'https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_infer.tar'
    }
    
    # 下载并解压每个模型
    for name, url in models.items():
        print(f"\nDownloading {name} model...")
        filename = f"{name}_model.tar"
        try:
            download_file(url, filename)
            print(f"Extracting {filename}...")
            extract_tar(filename)
            print(f"{name} model downloaded and extracted successfully!")
        except Exception as e:
            print(f"Error downloading {name} model: {str(e)}")

if __name__ == "__main__":
    main() 