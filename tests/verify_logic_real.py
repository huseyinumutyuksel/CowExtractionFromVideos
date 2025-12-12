import unittest
import numpy as np
import os
import shutil
import sys
import cv2

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.writer import CowVideoWriterManager
from src.processor import CowExtractionProcessor
from src.interfaces import IDetector
import config.settings as settings

class MockDetector(IDetector):
    def detect_and_track(self, frame):
        return []

class TestCowExtractionReal(unittest.TestCase):
    def setUp(self):
        self.output_dir = os.path.join(os.path.dirname(__file__), "test_output")
        if os.path.exists(self.output_dir):
            shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)
        
        # Override settings
        settings.OUTPUT_VIDEOS_DIR = self.output_dir
        # Set MIN duration to 0.5s for speed
        settings.MIN_TRACK_DURATION_SEC = 0.5
        settings.VIDEO_EXT = ".mp4"
        
    def tearDown(self):
        if os.path.exists(self.output_dir):
           shutil.rmtree(self.output_dir)

    def test_writer_manager_filtering(self):
        print("\nTesting WriterManager filtering...")
        manager = CowVideoWriterManager(self.output_dir)
        
        # FPS = 30
        fps = 30.0
        
        # Track 1: 5 frames (0.16s) -> Should be discarded (Min 0.5s)
        print("Writing Track 1 (Short)...")
        for i in range(5):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            manager.write_frame(1, frame, fps)
            
        # Track 2: 20 frames (0.66s) -> Should be saved
        print("Writing Track 2 (Long)...")
        for i in range(20):
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            manager.write_frame(2, frame, fps)
            
        manager.close_all()
        
        files = os.listdir(self.output_dir)
        print(f"Output files: {files}")
        
        # Expectation: Only 1 file "cow_0001.mp4"
        # Track 1 discarded. Track 2 saved.
        self.assertEqual(len(files), 1, "Should have exactly 1 output file")
        self.assertIn("cow_0001.mp4", files)
        
        # Verify duration of saved file?
        cap = cv2.VideoCapture(os.path.join(self.output_dir, "cow_0001.mp4"))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Saved video has {frame_count} frames")
        cap.release()
        self.assertEqual(frame_count, 20)

    def test_processor_logic_resize(self):
        print("\nTesting Processor resizing...")
        
        # Mock Detector
        class DummyResult:
            def __init__(self, boxes, ids):
                self.boxes = self
                self.xyxy = MockTensor(boxes)
                self.id = MockTensor(ids) if ids is not None else None
        
        class MockTensor:
            def __init__(self, data):
                self.data = np.array(data)
            def cpu(self): return self
            def numpy(self): return self.data
        
        detector = MockDetector()
        # Mock detect_and_track to return a box
        def side_effect(frame):
            # Return 1 cow
            # Box: 0,0, 50,50 (50x50 size)
            res = DummyResult([[0,0,50,50]], [1])
            return [res]
        
        detector.detect_and_track = side_effect
        
        manager = CowVideoWriterManager(self.output_dir)
        # Mock write_frame to check sizes
        from unittest.mock import MagicMock, patch
        
        manager.write_frame = MagicMock()
        
        processor = CowExtractionProcessor(detector, manager)
        
        # Patch VideoCapture to return 1 frame
        with patch('cv2.VideoCapture') as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            mock_cap.get.return_value = 30.0
            
            # Return 1 frame then empty
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_cap.read.side_effect = [(True, frame), (False, None)]
            
            mock_cap_cls.return_value = mock_cap
            
            # Set target resolution
            settings.OUTPUT_RESOLUTION = (64, 64)
            
            processor.process_video("dummy.mp4")
            
            # Check if write_frame called with resized frame
            self.assertTrue(manager.write_frame.called)
            args = manager.write_frame.call_args
            # args[0] is track_id, args[1] is frame
            written_frame = args[0][1]
            
            self.assertEqual(written_frame.shape, (64, 64, 3))
            print("Frame resized correctly to 64x64")

if __name__ == '__main__':
    unittest.main()
