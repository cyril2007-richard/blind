import cv2
import ncnn
import numpy as np
from collections import deque
import time
import matplotlib.pyplot as plt
from pathlib import Path


class YoloFastest:
    def __init__(self, param_path, bin_path, target_size=320, num_threads=4):
        self.target_size = target_size
        self.num_threads = num_threads
        self.mean_vals = [0, 0, 0]
        self.norm_vals = [1/255.0, 1/255.0, 1/255.0]
        self.debug_printed = False
        
        # COCO class names
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
            "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
            "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        
        # Pre-generate colors for all classes
        self.class_colors = {}
        for i in range(len(self.class_names)):
            np.random.seed(i)
            self.class_colors[i] = tuple(map(int, np.random.randint(50, 255, 3)))
        
        # Cache font parameters
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # Load model
        self.net = ncnn.Net()
        self.net.opt.use_vulkan_compute = True
        self.net.opt.num_threads = num_threads
        self.net.load_param(param_path)
        self.net.load_model(bin_path)
        
    def detect(self, img, conf_threshold=0.4):
        """Run detection on image"""
        img_h, img_w = img.shape[:2]
        
        # Preprocessing
        mat_in = ncnn.Mat.from_pixels_resize(
            img,
            ncnn.Mat.PixelType.PIXEL_BGR,
            img_w,
            img_h,
            self.target_size,
            self.target_size
        )
        mat_in.substract_mean_normalize(self.mean_vals, self.norm_vals)
        
        # Forward pass
        ex = self.net.create_extractor()
        ex.input("data", mat_in)
        ret, mat_out = ex.extract("output")
        
        # Debug: Print output shape once
        if not self.debug_printed:
            print(f"\n=== DEBUG INFO ===")
            print(f"Output dimensions: w={mat_out.w}, h={mat_out.h}, c={mat_out.c}")
            if mat_out.h > 0:
                sample_row = mat_out.row(0)
                print(f"Sample row length: {len(sample_row)}")
                print(f"Sample values: {sample_row[:min(10, len(sample_row))]}")
            print(f"==================\n")
            self.debug_printed = True
        
        # Parse detections
        proposals = []
        num_classes = len(self.class_names)
        
        for i in range(mat_out.h):
            values = mat_out.row(i)
            
            if len(values) < 6:
                continue
            
            class_id = int(values[0]) - 1
            confidence = values[1]
            
            if confidence <= conf_threshold or class_id < 0 or class_id >= num_classes:
                continue
            
            x1 = max(0, min(int(values[2] * img_w), img_w - 1))
            y1 = max(0, min(int(values[3] * img_h), img_h - 1))
            x2 = max(0, min(int(values[4] * img_w), img_w - 1))
            y2 = max(0, min(int(values[5] * img_h), img_h - 1))
            
            if x2 > x1 and y2 > y1:
                proposals.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': class_id
                })
        
        return proposals
    
    def draw_detections(self, img, detections):
        """Draw bounding boxes and labels on image"""
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_id = det['class_id']
            
            color = self.class_colors.get(class_id, (255, 255, 255))
            class_name = self.class_names[class_id]
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            
            label = f"{class_name}: {det['confidence']:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            
            y_label = max(y1 - 10, label_h + 10)
            cv2.rectangle(
                img,
                (x1, y_label - label_h - 10),
                (x1 + label_w + 10, y_label),
                color,
                -1
            )
            cv2.putText(
                img,
                label,
                (x1 + 5, y_label - 5),
                self.font,
                self.font_scale,
                (255, 255, 255),
                self.font_thickness
            )
        
        return img


class MetricsTracker:
    def __init__(self):
        self.response_times = []
        self.confidences = []
        self.frame_numbers = []
        self.detection_counts = []
        
    def add_metrics(self, frame_num, response_time, detections):
        self.frame_numbers.append(frame_num)
        self.response_times.append(response_time * 1000)  # Convert to ms
        self.detection_counts.append(len(detections))
        
        # Average confidence for this frame
        if detections:
            avg_conf = sum(d['confidence'] for d in detections) / len(detections)
            self.confidences.append(avg_conf)
        else:
            self.confidences.append(0)
    
    def plot_graphs(self, save_path):
        """Generate and save performance graphs"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('YOLO Detection Performance Metrics', fontsize=16, fontweight='bold')
        
        # Response Time Over Time
        axes[0, 0].plot(self.frame_numbers, self.response_times, color='blue', linewidth=1.5)
        axes[0, 0].set_title('Response Time per Frame', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Frame Number')
        axes[0, 0].set_ylabel('Response Time (ms)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Detection Count Over Time
        axes[0, 1].plot(self.frame_numbers, self.detection_counts, color='green', linewidth=1.5)
        axes[0, 1].set_title('Detections per Frame', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Frame Number')
        axes[0, 1].set_ylabel('Number of Detections')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Average Confidence Over Time
        axes[1, 0].plot(self.frame_numbers, self.confidences, color='orange', linewidth=1.5)
        axes[1, 0].set_title('Average Confidence per Frame', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Frame Number')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Summary Statistics
        avg_response = np.mean(self.response_times)
        avg_detections = np.mean(self.detection_counts)
        avg_confidence = np.mean([c for c in self.confidences if c > 0])
        
        stats_text = (
            f"Total Frames: {len(self.frame_numbers)}\n\n"
            f"Avg Response Time: {avg_response:.2f} ms\n"
            f"Min Response Time: {min(self.response_times):.2f} ms\n"
            f"Max Response Time: {max(self.response_times):.2f} ms\n\n"
            f"Avg Detections: {avg_detections:.2f}\n"
            f"Total Detections: {sum(self.detection_counts)}\n\n"
            f"Avg Confidence: {avg_confidence:.2f}"
        )
        
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        axes[1, 1].set_title('Summary Statistics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"\nGraphs saved to: {save_path}")


def main():
    # Initialize detector
    param_path = "blind-assistance-system/models/yolo-fastest-1.1.param"
    bin_path = "blind-assistance-system/models/yolo-fastest-1.1.bin"
    
    print("Loading YOLO-Fastest model...")
    detector = YoloFastest(param_path, bin_path, target_size=320)
    print("Model loaded successfully!")
    
    # Initialize metrics tracker
    metrics = MetricsTracker()
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting webcam detection. Press 'q' to quit.")
    print("Confidence threshold: 0.4\n")
    
    # For FPS calculation
    fps_buffer = deque(maxlen=30)
    prev_time = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        frame_count += 1
        
        # Measure detection time
        start_time = time.time()
        detections = detector.detect(frame, conf_threshold=0.4)
        detection_time = time.time() - start_time
        
        # Track metrics
        metrics.add_metrics(frame_count, detection_time, detections)
        
        # Simple print: Frame number and detections
        if detections:
            print(f"Frame {frame_count}:")
            for det in detections:
                class_name = detector.class_names[det['class_id']]
                print(f"  {class_name} - Confidence: {det['confidence']:.3f}")
        
        # Draw results
        frame = detector.draw_detections(frame, detections)
        
        # Calculate FPS
        curr_time = cv2.getTickCount()
        if prev_time != 0:
            fps_buffer.append(cv2.getTickFrequency() / (curr_time - prev_time))
        prev_time = curr_time
        fps = sum(fps_buffer) / len(fps_buffer) if fps_buffer else 0
        
        # Display info
        info_text = f"FPS: {fps:.1f} | Detections: {len(detections)} | Time: {detection_time*1000:.1f}ms"
        cv2.rectangle(frame, (5, 5), (550, 45), (0, 0, 0), -1)
        cv2.putText(
            frame,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Show frame
        cv2.imshow("YOLO-Fastest Detection", frame)
        
        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Ask for graph filename and save
    print("\n" + "="*50)
    print("Detection session completed!")
    print(f"Total frames processed: {frame_count}")
    print("="*50)
    
    if frame_count > 0:
        graph_name = input("\nEnter filename for the graph (without extension): ").strip()
        if not graph_name:
            graph_name = f"detection_metrics_{int(time.time())}"
        
        save_path = logs_dir / f"{graph_name}.png"
        metrics.plot_graphs(save_path)
        print(f"\nSession complete. Logs saved in '{logs_dir}' directory.")
    else:
        print("No frames processed, no graphs to save.")


if __name__ == "__main__":
    main()