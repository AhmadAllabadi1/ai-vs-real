import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
import json
from tqdm import tqdm

class ImagePreprocessor:
    
    def __init__(self, 
                 source_dir='Dataset',
                 output_dir='Dataset_Preprocessed',
                 target_size=(224, 224),
                 sample_size=5000,
                 random_seed=42):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.target_size = target_size
        self.sample_size = sample_size
        self.random_seed = random_seed
        
        np.random.seed(random_seed)
        
        self.stats = {
            'train': {'real': [], 'fake': []},
            'test': {'real': [], 'fake': []}
        }
        
    def analyze_dataset(self):
        print("=" * 60)
        print("DATASET ANALYSIS")
        print("=" * 60)
        
        analysis = {}
        
        for split in ['train', 'test']:
            analysis[split] = {}
            for class_name in ['fake', 'real']:
                path = self.source_dir / split / class_name
                if not path.exists():
                    print(f"Warning: {path} does not exist!")
                    continue
                
                images = list(path.glob('*'))
                images = [img for img in images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
                
                sizes = []
                file_sizes = []
                aspect_ratios = []
                
                print(f"\nAnalyzing {split}/{class_name}...")
                for img_path in tqdm(images[:100], desc=f"Sampling {split}/{class_name}"):
                    try:
                        img = Image.open(img_path)
                        sizes.append(img.size)
                        file_sizes.append(img_path.stat().st_size / 1024)  # KB
                        aspect_ratios.append(img.size[0] / img.size[1])
                        img.close()
                    except Exception as e:
                        print(f"Error reading {img_path}: {e}")
                
                analysis[split][class_name] = {
                    'count': len(images),
                    'avg_width': np.mean([s[0] for s in sizes]),
                    'avg_height': np.mean([s[1] for s in sizes]),
                    'min_width': np.min([s[0] for s in sizes]),
                    'max_width': np.max([s[0] for s in sizes]),
                    'min_height': np.min([s[1] for s in sizes]),
                    'max_height': np.max([s[1] for s in sizes]),
                    'avg_file_size_kb': np.mean(file_sizes),
                    'avg_aspect_ratio': np.mean(aspect_ratios)
                }
        
        # Print analysis
        for split in ['train', 'test']:
            print(f"\n{split.upper()} SET:")
            for class_name in ['real', 'fake']:
                if class_name in analysis[split]:
                    stats = analysis[split][class_name]
                    print(f"  {class_name.upper()}:")
                    print(f"    Total images: {stats['count']}")
                    print(f"    Avg dimensions: {stats['avg_width']:.0f} x {stats['avg_height']:.0f}")
                    print(f"    Size range: ({stats['min_width']:.0f}, {stats['min_height']:.0f}) to ({stats['max_width']:.0f}, {stats['max_height']:.0f})")
                    print(f"    Avg file size: {stats['avg_file_size_kb']:.2f} KB")
                    print(f"    Avg aspect ratio: {stats['avg_aspect_ratio']:.2f}")
        
        return analysis
    
    def files(self, split='train'):
        files = {}
        
        for class_name in ['real', 'fake']:
            path = self.source_dir / split / class_name
            all_images = list(path.glob('*'))
            all_images = [img for img in all_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
            files[class_name]  = all_images
            print(f"{len(files)} images from {split}/{class_name}")  
        return files
    
    def preprocess_image(self, img_path, output_path):
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_resized = img.resize(self.target_size, Image.Resampling.LANCZOS)
            img_resized.save(output_path, 'JPEG', quality=95)
            
            img_array = np.array(img_resized)
            return {
                'mean': img_array.mean(),
                'std': img_array.std(),
                'min': img_array.min(),
                'max': img_array.max()
            }
            
        except Exception as e:
            print(f"Error preprocessing {img_path}: {e}")
            return None
    
    def preprocess_dataset(self):
        for split in ['train', 'test']:
            for class_name in ['real', 'fake']:
                output_path = self.output_dir / split / class_name
                output_path.mkdir(parents=True, exist_ok=True)
        
        for split in ['train', 'test']:
            print(f"\nProcessing {split.upper()} set...")
            
            if split == 'train':
                files = self.files(split)
            else:
                files = {}
                for class_name in ['real', 'fake']:
                    path = self.source_dir / split / class_name
                    all_images = list(path.glob('*'))
                    all_images = [img for img in all_images if img.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
                    files[class_name] = all_images
            
            for class_name in ['real', 'fake']:
                print(f"\n  Processing {class_name} images...")
                
                for i, img_path in enumerate(tqdm(files[class_name])):
                    output_path = self.output_dir / split / class_name / f"{class_name}_{i:05d}.jpg"
                    
                    stats = self.preprocess_image(img_path, output_path)
                    
                    if stats:
                        self.stats[split][class_name].append(stats)
        self.save_statistics()
        
        print("\n" + "=" * 60)
        print("PREPROCESSING COMPLETE")
        print("=" * 60)
        self.print_statistics()
    
    def save_statistics(self):
        stats_summary = {}
        
        for split in ['train', 'test']:
            stats_summary[split] = {}
            for class_name in ['real', 'fake']:
                if self.stats[split][class_name]:
                    all_means = [s['mean'] for s in self.stats[split][class_name]]
                    all_stds = [s['std'] for s in self.stats[split][class_name]]
                    
                    stats_summary[split][class_name] = {
                        'count': len(self.stats[split][class_name]),
                        'avg_pixel_mean': float(np.mean(all_means)),
                        'avg_pixel_std': float(np.mean(all_stds)),
                        'target_size': self.target_size
                    }
        
        stats_path = self.output_dir / 'preprocessing_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats_summary, f, indent=2)
        
        print(f"\nStatistics saved to {stats_path}")
    
    def print_statistics(self):
        print("\nPREPROCESSED DATASET STATISTICS:")
        print(f"Target size: {self.target_size}")
        
        for split in ['train', 'test']:
            print(f"\n{split.upper()} SET:")
            for class_name in ['real', 'fake']:
                if self.stats[split][class_name]:
                    all_means = [s['mean'] for s in self.stats[split][class_name]]
                    all_stds = [s['std'] for s in self.stats[split][class_name]]
                    
                    print(f"  {class_name.upper()}:")
                    print(f"    Count: {len(self.stats[split][class_name])}")
                    print(f"    Avg pixel mean: {np.mean(all_means):.2f}")
                    print(f"    Avg pixel std: {np.mean(all_stds):.2f}")


def main():
    print("Image Preprocessing for Real vs AI-Generated Classification")
    print("=" * 60)
    
    preprocessor = ImagePreprocessor(
        source_dir='Dataset',
        output_dir='data',
        target_size=(224, 224),
        sample_size=60000,
        random_seed=42
    )
    
    analysis = preprocessor.analyze_dataset()
    preprocessor.preprocess_dataset()
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print(f"Preprocessed dataset saved to: {preprocessor.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()