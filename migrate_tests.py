#!/usr/bin/env python3
"""
Migration script to move old test files to new package structure
"""

import os
import shutil
from pathlib import Path

def migrate_tests():
    """Migrate old test files to new structure"""
    
    print("🔄 Migrating Test Files to Package Structure")
    print("=" * 50)
    
    # Create test outputs directory
    test_outputs = Path("test_outputs")
    test_outputs.mkdir(exist_ok=True)
    
    # Move old test outputs to new structure
    old_outputs = [
        "video_inference_output",
        "trained_model_inference_output", 
        "pretrained_vgg11_output",
        "pretrained_efficientnet_output"
    ]
    
    for old_output in old_outputs:
        if os.path.exists(old_output):
            new_path = test_outputs / old_output
            if not new_path.exists():
                shutil.move(old_output, new_path)
                print(f"✅ Moved {old_output} -> test_outputs/{old_output}")
            else:
                print(f"⚠️  {old_output} already exists in test_outputs/")
    
    # Create symlinks for backward compatibility
    print("\n🔗 Creating backward compatibility symlinks...")
    
    for old_output in old_outputs:
        if not os.path.exists(old_output) and (test_outputs / old_output).exists():
            os.symlink(f"test_outputs/{old_output}", old_output)
            print(f"✅ Created symlink: {old_output} -> test_outputs/{old_output}")
    
    # Update old test scripts to use new package
    print("\n📝 Updating old test scripts...")
    
    # Update simple_video_inference_test.py
    if os.path.exists("simple_video_inference_test.py"):
        with open("simple_video_inference_test.py", "r") as f:
            content = f.read()
        
        # Add import and usage of new package
        new_content = content.replace(
            "def main():",
            '''def main():
    # Use new test package
    from tests.test_video_inference import VideoInferenceTester
    
    tester = VideoInferenceTester()
    video_path = "video_processing/data3_users_yoavnavon_clips_dqa_glued_sv_vehicle_day_FRONT_RIGHT_GENERIC_1x.mp4"
    
    # Run tests using new package
    tester.test_dummy_inference(video_path, max_frames=10)
    return
    
    # Original code below (commented out)
    # '''
        )
        
        with open("simple_video_inference_test.py", "w") as f:
            f.write(new_content)
        
        print("✅ Updated simple_video_inference_test.py")
    
    print("\n✅ Migration completed!")
    print("\n📁 New structure:")
    print("  tests/")
    print("  ├── __init__.py")
    print("  ├── test_video_inference.py")
    print("  ├── test_model_comparison.py")
    print("  ├── test_training.py")
    print("  └── run_tests.py")
    print("  test_outputs/")
    print("  └── (all test outputs)")
    
    print("\n🚀 Usage:")
    print("  python tests/run_tests.py --help")
    print("  python tests/run_tests.py --tests inference")
    print("  python tests/run_tests.py --tests all --max-frames 10")

if __name__ == "__main__":
    migrate_tests()
