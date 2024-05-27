import numpy as np
import cv2

def apply_custom_filter(matrix, output_path):
    diag_kernel = np.array([[1, -1, -1],
                            [-1, 1, -1],
                            [-1, -1, 1]])

    filtered_matrix = cv2.filter2D(matrix, -1, diag_kernel)

    normalized_matrix = cv2.normalize(filtered_matrix, None, 0, 127, cv2.NORM_MINMAX)

    threshold_level = 50
    _, binary_matrix = cv2.threshold(normalized_matrix, threshold_level, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_path, binary_matrix)
    cv2.imshow('Filtered Dotplot', binary_matrix)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
