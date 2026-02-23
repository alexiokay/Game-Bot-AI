
import cv2
import numpy as np

print("Testing cv2 build...")
try:
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    # We won't actually show window to block, just check if function exists/runs
    # actually imshow might block or need waitKey.
    # Just checking if calling it throws "Not Implemented"
    # We will wrap it in try/except and print result
    
    cv2.imshow("Test", img)
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    print("SUCCESS: cv2.imshow worked")
except cv2.error as e:
    print(f"FAILURE: cv2 error: {e}")
except Exception as e:
    print(f"FAILURE: other error: {e}")
