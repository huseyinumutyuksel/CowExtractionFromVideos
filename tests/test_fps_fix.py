import unittest
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.processor import CowExtractionProcessor
import config.settings as settings

class TestFpsFix(unittest.TestCase):
    def test_fps_rounding(self):
        print("\nTesting FPS rounding fix...")
        
        # Mock dependencies
        mock_detector = MagicMock()
        mock_writer_manager = MagicMock()
        
        processor = CowExtractionProcessor(mock_detector, mock_writer_manager)
        
        # Patch VideoCapture
        with patch('cv2.VideoCapture') as mock_cap_cls:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = True
            
            # Simulate the problematic FPS
            # 240373/1000 = 240.373 which caused the error
            problematic_fps = 240.373
            mock_cap.get.return_value = problematic_fps
            
            # Return 1 frame then empty
            frame = np.zeros((100, 100, 3), dtype=np.uint8)
            mock_cap.read.side_effect = [(True, frame), (False, None)]
            
            mock_cap_cls.return_value = mock_cap
            
            # Mock detector to find a cow so write_frame is called
            class DummyResult:
                def __init__(self):
                    self.boxes = self
                    self.xyxy = MagicMock()
                    self.xyxy.cpu().numpy().astype.return_value = [[0,0,50,50]]
                    self.id = MagicMock()
                    self.id.cpu().numpy().astype.return_value = [1]
            
            mock_detector.detect_and_track.return_value = [DummyResult()]
            
            # Run
            processor.process_video("dummy_path.mp4")
            
            # Verification
            self.assertTrue(mock_writer_manager.write_frame.called)
            args = mock_writer_manager.write_frame.call_args
            # args: (track_id, frame, fps)
            passed_fps = args[0][2]
            
            print(f"Original FPS: {problematic_fps}")
            print(f"Passed FPS: {passed_fps}")
            
            self.assertIsInstance(passed_fps, int, "FPS should be an integer")
            self.assertEqual(passed_fps, 240, "FPS should be rounded to 240")

if __name__ == '__main__':
    unittest.main()
