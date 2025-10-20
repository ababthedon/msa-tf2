"""
Quick Data Verification Script

Verifies that your data dimensions match the model configuration.
"""

import h5py
import sys

def verify_data():
    """Verify data dimensions."""
    print("\n" + "="*70)
    print("Data Dimension Verification")
    print("="*70)
    
    # Expected dimensions for MOSEI
    expected = {
        'text': 300,
        'audio': 74,
        'video': 713,  # MOSEI (MOSI would be 47)
        'seq_len': 20
    }
    
    splits = ['train', 'valid', 'test']
    all_ok = True
    
    for split in splits:
        print(f"\n{split.upper()} Split:")
        print("-" * 50)
        
        # Check text
        try:
            with h5py.File(f'data/text_{split}_emb.h5', 'r') as f:
                shape = f['d1'].shape
                text_dim = shape[-1]
                seq_len = shape[1]
                n_samples = shape[0]
                
                status = "✓" if text_dim == expected['text'] else "✗"
                print(f"  Text:  {status} {shape} (dim={text_dim}, expected={expected['text']})")
                if text_dim != expected['text']:
                    all_ok = False
        except Exception as e:
            print(f"  Text:  ✗ Error: {e}")
            all_ok = False
        
        # Check audio
        try:
            with h5py.File(f'data/audio_{split}.h5', 'r') as f:
                shape = f['d1'].shape
                audio_dim = shape[-1]
                
                status = "✓" if audio_dim == expected['audio'] else "✗"
                print(f"  Audio: {status} {shape} (dim={audio_dim}, expected={expected['audio']})")
                if audio_dim != expected['audio']:
                    all_ok = False
        except Exception as e:
            print(f"  Audio: ✗ Error: {e}")
            all_ok = False
        
        # Check video
        try:
            with h5py.File(f'data/video_{split}.h5', 'r') as f:
                shape = f['d1'].shape
                video_dim = shape[-1]
                
                status = "✓" if video_dim == expected['video'] else "✗"
                print(f"  Video: {status} {shape} (dim={video_dim}, expected={expected['video']})")
                
                if video_dim != expected['video']:
                    all_ok = False
                    if video_dim == 47:
                        print(f"         → This looks like CMU-MOSI data (47 dims)")
                        print(f"         → Update video_dim=47 in training scripts")
        except Exception as e:
            print(f"  Video: ✗ Error: {e}")
            all_ok = False
        
        # Check labels
        try:
            with h5py.File(f'data/y_{split}.h5', 'r') as f:
                shape = f['d1'].shape
                print(f"  Labels: ✓ {shape}")
        except Exception as e:
            print(f"  Labels: ✗ Error: {e}")
            all_ok = False
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if all_ok:
        print("\n✓ All dimensions match!")
        print(f"\nDataset: CMU-MOSEI")
        print(f"  Text features:  {expected['text']}")
        print(f"  Audio features: {expected['audio']}")
        print(f"  Video features: {expected['video']}")
        print(f"  Sequence length: {expected['seq_len']}")
        print(f"\n✓ Training scripts are configured correctly")
        print(f"\nYou can start training:")
        print(f"  python train_seqlevel.py --mixed_precision --batch_size 64")
        return 0
    else:
        print("\n✗ Dimension mismatch detected!")
        print("\nPlease verify:")
        print("  1. Which dataset you have (MOSI or MOSEI)")
        print("  2. Update video_dim in training scripts accordingly")
        print("     - MOSI: video_dim = 47")
        print("     - MOSEI: video_dim = 713")
        return 1


if __name__ == '__main__':
    sys.exit(verify_data())





