import cv2
import numpy as np
import json
import argparse

# Global variables to store selected points
selected_points = []

def undistort_image(frame, mtx, dist):
    """
    Undistort a frame using camera calibration parameters.

    Args:
        frame: Input frame (image) from the camera.
        mtx: Camera matrix.
        dist: Distortion coefficients.

    Returns:
        dst: The undistorted image.
    """

    # Get size
    h,  w = frame.shape[:2]

    # Get optimal new camera
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

    # Undistort frame
    dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)

    # Crop image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    return dst


def compute_line_segments(points, Z):
    """
    Compute the perimeter of the object.

    Args:
        distances (list): List of distances between consecutive points.

    Returns:
        float: The perimeter of the object.
    """

    # Inicializar la lista de distancias ajustadas
    adjusted_distances = []

    # Calcular las distancias entre puntos consecutivos
    for i in range(len(points) - 1):
        # Calcular la distancia euclidiana entre los puntos
        dist = np.linalg.norm(np.array(points[i]) - np.array(points[i+1]))
        # Ajustar la distancia según la distancia Z
        adjusted_dist = dist * Z *2 /1000
        adjusted_distances.append(adjusted_dist)

    # Calcular la distancia entre el último punto y el primero para cerrar la forma
    dist = np.linalg.norm(np.array(points[-1]) - np.array(points[0]))
    adjusted_dist = dist * Z * 2 /1000
    adjusted_distances.append(adjusted_dist)
    
    return adjusted_distances




def compute_perimeter(distances):
    """
    Compute the perimeter of the object.

    Args:
        distances (list): List of distances between consecutive points.

    Returns:
        float: The perimeter of the object.
    """

    # Compute the perimeter by summing all distances
    perimeter = sum(distances)
    return perimeter

def mouse_callback(event, x, y, flags, params):
    """
    Callback function for mouse events.

    Args:
        event (int): Type of mouse event.
        x (int): x-coordinate of the mouse cursor.
        y (int): y-coordinate of the mouse cursor.
        flags (int): Additional flags.
        params (object): Additional parameters.
    """

    global selected_points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        # Left mouse button is pressed, store the point
        selected_points.append((x, y))
        print("Point selected: ({}, {})".format(x, y))
    
    elif event == cv2.EVENT_MBUTTONDOWN:
        # Middle mouse button is pressed, stop selecting points
        print("Selection stopped.")
        cv2.setMouseCallback('Result Frame', lambda *args : None)
    
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right mouse button is pressed, clear selected points
        selected_points = []
        print("Points cleared.")

    elif event == cv2.EVENT_LBUTTONUP:
        # Left mouse button is released
        pass

def main(cam_index, Z, cal_file):
     """
    Main function to perform vision-based object measurement.

    Args:
        cam_index (int): Index of the camera.
        Z (int): Distance from camera to object.
        cal_file (str): Path to the calibration file.
    """
    # Load calibration data
    with open(cal_file, 'r') as f:
        calibration_data = json.load(f)
    
    # Extract camera matrix and distortion coefficients from calibration data
    mtx = np.array(calibration_data['camera_matrix'])
    dist = np.array(calibration_data['distortion_coefficients'])
    
    # Initialize camera
    cap = cv2.VideoCapture(cam_index)
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        if not ret:
            break
        
        # Undistort frame
        undistorted_frame = undistort_image(frame, mtx, dist)
        
        # Display undistorted frame
        cv2.imshow('Undistorted Frame', undistorted_frame)
        
        # Set mouse callback
        cv2.setMouseCallback('Undistorted Frame', mouse_callback)
        
        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if len(selected_points) > 0:
            # Dibujar los puntos y las líneas en la imagen
            result_frame = undistorted_frame.copy()
            for point in selected_points:
                cv2.circle(result_frame, point, 5, (0, 255, 0), -1)
            for i in range(len(selected_points) - 1):
                cv2.line(result_frame, selected_points[i], selected_points[i+1], (255, 0, 0), 2)
            cv2.line(result_frame, selected_points[-1], selected_points[0], (255, 0, 0), 2)  # Conectar el último punto con el primero
            cv2.imshow('Result Frame', result_frame)

            # Limpiar los puntos seleccionados para la próxima medición
            if cv2.waitKey(1) & 0xFF == ord('c'):
                selected_points.clear()
    
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()

    if len(selected_points) < 3:
        print("At least 3 points are required to calculate the perimeter.")
        return

    # Compute distances between selected points
    distances = compute_line_segments(selected_points, Z)
    
    # Compute perimeter
    perimeter = compute_perimeter(distances)

    # Print results
    print("Distance between consecutive points:")
    for i, dist in enumerate(distances):
        print(f"P{i} to P{i+1}: {dist}")
    
    print("Perimeter:", perimeter)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vision-based object measurement')
    parser.add_argument('--cam_index', type=int, default=0, help='Index of the camera')
    parser.add_argument('--Z', type=int, help='Distance from camera to object')
    parser.add_argument('--cal_file', type=str, default='calibration_data.json', help='Calibration file')
    args = parser.parse_args()
    
    main(args.cam_index, args.Z, args.cal_file)
