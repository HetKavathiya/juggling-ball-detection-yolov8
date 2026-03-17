"""
Main entry point - Run complete juggling ball analysis
"""
from process_juggling_video import JugglingAnalysisPipeline
import os

def main():
    print("\n" + "="*70)
    print("🎪 JUGGLING BALL DETECTION & TRACKING SYSTEM")
    print("="*70 + "\n")
    
    # Configuration
    MODEL_PATH = 'runs/detect/juggling_ball_detector/weights/best.pt'
    NUM_BALLS = 3  # Change this based on your video
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model not found: {MODEL_PATH}")
        print("Please make sure training is completed first!")
        return
    
    # Initialize pipeline
    pipeline = JugglingAnalysisPipeline(
        model_path=MODEL_PATH,
        num_balls=NUM_BALLS,
        confidence_threshold=0.45,
        max_tracking_distance=80
    )
    
    # Menu
    print("Select operation:")
    print("1. Process VIDEO (detects balls + shows paths)")
    print("2. Process IMAGE (detects balls only)")
    print("3. Process both VIDEO and IMAGE")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == '1':
        # Video processing
        video_path = input("\nEnter video path (e.g., input_videos/juggling.mp4): ").strip()
        output_path = input("Enter output path (e.g., output_videos/juggling_analyzed.mp4): ").strip()
        
        if not os.path.exists(video_path):
            print(f"❌ Video not found: {video_path}")
            return
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        pipeline.process_video(
            input_path=video_path,
            output_path=output_path,
            display=True,
            save_stats=True
        )
    
    elif choice == '2':
        # Image processing
        image_path = input("\nEnter image path (e.g., input_images/juggling.jpg): ").strip()
        output_path = input("Enter output path (e.g., output_images/juggling_analyzed.jpg): ").strip()
        
        if not os.path.exists(image_path):
            print(f"❌ Image not found: {image_path}")
            return
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        
        result = pipeline.process_image(
            image_path=image_path,
            output_path=output_path
        )
        
        if result is not None:
            print("✅ Image processing completed!")
    
    elif choice == '3':
        # Process both
        video_path = input("\nEnter video path: ").strip()
        video_output = input("Enter video output path: ").strip()
        image_path = input("Enter image path: ").strip()
        image_output = input("Enter image output path: ").strip()
        
        if os.path.exists(video_path):
            os.makedirs(os.path.dirname(video_output) or '.', exist_ok=True)
            pipeline.process_video(video_path, video_output, display=True, save_stats=True)
        
        if os.path.exists(image_path):
            os.makedirs(os.path.dirname(image_output) or '.', exist_ok=True)
            pipeline.process_image(image_path, image_output)
    
    else:
        print("Exiting...")

if __name__ == "__main__":
    main()