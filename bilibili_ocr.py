import yt_dlp
import cv2
import numpy as np
from paddleocr import PaddleOCR
import os
import tempfile
import glob
import shutil
import argparse

def cleanup_temp_files(file_patterns=None):
    """清理临时文件"""
    if file_patterns is None:
        file_patterns = ['*.mp4', '*.part', '*.ytdl', '*.jpg', '*.png', '*.webm', '*.m4a', '*_cache*']
    
    temp_dir = tempfile.gettempdir()
    cleaned_files = []
    
    # 清理临时目录中的文件
    for pattern in file_patterns:
        pattern_path = os.path.join(temp_dir, pattern)
        for file_path in glob.glob(pattern_path):
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    cleaned_files.append(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
                    cleaned_files.append(file_path)
            except Exception as e:
                print(f"无法删除 {file_path}: {str(e)}")
    
    # 清理可能存在的bilibili_ocr_temp目录
    bilibili_temp = os.path.join(temp_dir, 'bilibili_ocr_temp')
    if os.path.exists(bilibili_temp):
        try:
            shutil.rmtree(bilibili_temp)
            cleaned_files.append(bilibili_temp)
        except Exception as e:
            print(f"无法删除临时目录 {bilibili_temp}: {str(e)}")
    
    if cleaned_files:
        print(f"已清理 {len(cleaned_files)} 个临时文件/目录")
        for file in cleaned_files:
            print(f"- 已删除: {os.path.basename(file)}")

def download_video(url):
    """Download video from Bilibili using yt-dlp"""
    temp_dir = tempfile.gettempdir()
    temp_video_dir = os.path.join(temp_dir, 'bilibili_ocr_temp')
    os.makedirs(temp_video_dir, exist_ok=True)

    ydl_opts = {
        'format': 'mp4',
        'outtmpl': os.path.join(temp_video_dir, '%(id)s.%(ext)s'),
        'quiet': False,
        'no_warnings': False,
        'force_generic_extractor': False,
        # 添加更多的下载选项
        'socket_timeout': 30,
        'retries': 3,
        'fragment_retries': 3,
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36',
            'Referer': 'https://www.bilibili.com'
        }
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"正在解析视频地址: {url}")
            try:
                info = ydl.extract_info(url, download=False)
                if not info:
                    raise Exception("无法获取视频信息")
                
                print(f"视频标题: {info.get('title', 'Unknown')}")
                print(f"视频ID: {info.get('id', 'Unknown')}")
                
                formats = info.get('formats', [])
                if not formats:
                    raise Exception("没有找到可用的视频格式")
                
                print(f"可用的视频格式数量: {len(formats)}")
                
                # 选择最佳的MP4格式
                mp4_formats = []
                for f in formats:
                    if f.get('ext') == 'mp4':
                        mp4_formats.append(f)
                
                if not mp4_formats:
                    raise Exception("未找到MP4格式的视频")
                
                # 优先选择有filesize信息的格式
                selected_format = None
                for f in mp4_formats:
                    if f.get('filesize'):
                        if not selected_format or f.get('filesize', 0) > selected_format.get('filesize', 0):
                            selected_format = f
                
                # 如果没有找到带filesize的格式，就选择第一个MP4格式
                if not selected_format:
                    selected_format = mp4_formats[0]
                
                print(f"选择的格式: {selected_format.get('format_id')} ({selected_format.get('ext')})")
                if selected_format.get('filesize'):
                    print(f"预计文件大小: {selected_format.get('filesize')} bytes")
                else:
                    print("文件大小信息不可用")
                
                ydl_opts['format'] = selected_format['format_id']
                print(f"开始下载视频...")
                
                # 重新下载指定格式
                info = ydl.extract_info(url, download=True)
                video_path = ydl.prepare_filename(info)
                
                if not os.path.exists(video_path):
                    raise Exception(f"下载完成但文件不存在: {video_path}")
                    
                file_size = os.path.getsize(video_path)
                print(f"视频下载完成: {video_path}")
                print(f"实际文件大小: {file_size} bytes")
                
                return video_path
                    
            except yt_dlp.utils.DownloadError as e:
                print(f"yt-dlp下载错误: {str(e)}")
                raise
            except Exception as e:
                print(f"视频信息获取失败: {str(e)}")
                raise
                
    except Exception as e:
        print(f"下载视频时出错: {str(e)}")
        raise

def extract_key_frames(video_path, interval=30):
    """Extract key frames from video at specified interval"""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
    frames = []
    cap = None
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("无法打开视频文件")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"视频总帧数: {total_frames}")
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % interval == 0:
                frames.append(frame)
                print(f"提取关键帧: {len(frames)}/{total_frames//interval}", end='\r')
        
        print(f"\n成功提取 {len(frames)} 个关键帧")
        return frames
        
    except Exception as e:
        print(f"提取关键帧时出错: {str(e)}")
        raise
        
    finally:
        if cap is not None:
            cap.release()
            cv2.destroyAllWindows()

def perform_ocr(frames):
    """Perform OCR on frames using PaddleOCR"""
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang="ch",
        # 基础配置
        use_gpu=False,
        enable_mkldnn=False,
        # 检测模型配置
        det_db_thresh=0.3,
        det_db_box_thresh=0.5,
        det_db_unclip_ratio=1.6,
        use_dilation=False,
        det_db_score_mode="fast",
        # 内存和计算优化
        rec_batch_num=1,
        cls_batch_num=1,
        max_batch_size=1,
        use_mp=False,
        total_process_num=1,
        cpu_threads=2,
        # 禁用不必要的模型
        rec_algorithm='CRNN',
        det_algorithm='DB'
    )
    results = []
    
    for frame in frames:
        try:
            result = ocr.ocr(frame, cls=True)
            if result[0] is not None:
                results.append(result[0])
        except Exception as e:
            print(f"Warning: OCR failed for a frame: {str(e)}")
            continue
    
    return results

def main(url):
    print(f"开始处理视频: {url}")
    video_path = None
    temp_video_dir = os.path.join(tempfile.gettempdir(), 'bilibili_ocr_temp')
    
    try:
        # 确保临时目录存在
        os.makedirs(temp_video_dir, exist_ok=True)
        
        # 下载视频
        video_path = download_video(url)
        if not video_path or not os.path.exists(video_path):
            raise Exception("视频下载失败")
        
        print("正在提取关键帧...")
        frames = extract_key_frames(video_path)
        
        if not frames:
            print("警告: 没有提取到任何帧!")
            return
            
        print(f"提取到 {len(frames)} 个关键帧")
        print("正在进行OCR识别...")
        ocr_results = perform_ocr(frames)
        
        # 处理OCR结果...
        process_ocr_results(ocr_results)
        
        print("所有处理已完成！")
        
    except Exception as e:
        print(f"处理过程中出错: {str(e)}")
        raise
    
    finally:
        # 所有处理完成后，清理临时文件
        try:
            if video_path and os.path.exists(video_path):
                os.remove(video_path)
                print(f"已删除视频文件: {os.path.basename(video_path)}")
            
            if os.path.exists(temp_video_dir):
                shutil.rmtree(temp_video_dir)
                print(f"已清理临时目录: {temp_video_dir}")
            
            # 清理其他临时文件
            cleanup_temp_files()
            
        except Exception as e:
            print(f"清理临时文件时出错: {str(e)}")

def process_ocr_results(ocr_results):
    """处理OCR结果并显示"""
    unique_texts = {}
    
    for frame_results in ocr_results:
        if not frame_results:
            continue
        frame_text = []
        for line in frame_results:
            text = line[1][0]
            confidence = line[1][1]
            if text not in ["bilibili", "诗辞人间"]:
                frame_text.append(text)
        
        frame_key = "\n".join(sorted(frame_text))
        if frame_key and frame_key not in unique_texts:
            unique_texts[frame_key] = frame_text
    
    print("\n=== 视频中的唯一文案 ===")
    if not unique_texts:
        print("没有识别到任何文字!")
    else:
        for i, texts in enumerate(unique_texts.values(), 1):
            print(f"\n文案组 {i}:")
            for text in texts:
                print(f"- {text}")
            print("-" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bilibili视频OCR处理工具')
    parser.add_argument('url', help='Bilibili视频URL')
    args = parser.parse_args()
    
    main(args.url) 