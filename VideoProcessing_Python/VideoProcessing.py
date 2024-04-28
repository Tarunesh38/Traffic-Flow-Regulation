# importing libraries for numpy and opencv for reading the video and take the frames
# also importing library to do blob analysis
import numpy as np
import cv2

# Function to convery image to grayscale
def grayscale(frame):
    # Grayscale conversion from scratch
    image_height, image_width = frame.shape[0], frame.shape[1]
    output_image = np.zeros((image_height, image_width))
    for i in range(image_height):
        for j in range(image_width):
            output_image[i, j] = 0.299 * frame[i, j, 0] + 0.587 * frame[i, j, 1] + 0.114 * frame[i, j, 2]
    return output_image

# Function to perform 2D Convolution from scratch
def convolution(image, kernel):
    # Get the dimensions of the image and the kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    # Calculate the padding required for the convolution operation
    padding_height = kernel_height // 2
    padding_width = kernel_width // 2
    # Create a padded image with zeros
    padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')
    # Create an empty output image
    output_image = np.zeros_like(image)
    # Perform the convolution operation
    for i in range(image_height):
        for j in range(image_width):
            output_image[i, j] = np.sum(padded_image[i:i+kernel_height, j:j+kernel_width] * kernel)
    return output_image

# Function to perform morphological operations
def morph(image, structure, type):
    if type == "erosion":
        # Erosion from scratch
        image_height, image_width = image.shape
        structure_height, structure_width = structure.shape
        padding_height = structure_height // 2
        padding_width = structure_width // 2

        padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')

        output_image = np.zeros_like(image)

        for i in range(image_height):
            for j in range(image_width):
                output_image[i, j] = np.min(padded_image[i:i+structure_height, j:j+structure_width] * structure)
        return output_image
    elif type == "dilation":
        # Dilation from scratch
        image_height, image_width = image.shape
        structure_height, structure_width = structure.shape
        padding_height = structure_height // 2
        padding_width = structure_width // 2

        padded_image = np.pad(image, ((padding_height, padding_height), (padding_width, padding_width)), mode='constant')

        output_image = np.zeros_like(image)

        for i in range(image_height):
            for j in range(image_width):
                output_image[i, j] = np.max(padded_image[i:i+structure_height, j:j+structure_width] * structure)
        return output_image
    elif type == "opening":
        # Opening from scratch
        erosion = morph(image, structure, "erosion")
        dilation = morph(erosion, structure, "dilation")
        return dilation
    elif type == "closing":
        # Closing from scratch
        dilation = morph(image, structure, "dilation")
        erosion = morph(dilation, structure, "erosion")
        return erosion
    
# Function to perform Binary conversion
def binary_conversion(image, threshold):
    # Binary conversion from scratch
    image_height, image_width = image.shape
    output_image = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            if image[i, j] > threshold:
                output_image[i, j] = 255
            else:
                output_image[i, j] = 0
    return output_image

# Function to perform Blob analysis and return the number of blobs detected
def blob_analysis(frame):
    # Convert the frame to a format accepted by cv2.findContours()
    frame = frame.astype(np.uint8)

    # Find the contours
    contours, hierarchy = cv2.findContours(frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter blobs based on minimum area
    min_blob_area = 150
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_blob_area]

    # Draw bounding boxes on the original frame
    for contour in filtered_contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Count the number of detected blobs
    num_blobs = len(filtered_contours)
    
    return num_blobs, frame

# Structuring Element
structure = np.array([[1, 1, 1],
                      [1, 1, 1],
                      [1, 1, 1]]) # Square Structuring Element for Opening operation

# Define the Prewitt kernel for edge detection
prewitt_kernel_x = np.array([[-1, 0, 1],
                             [-1, 0, 1],
                             [-1, 0, 1]])

prewitt_kernel_y = np.array([[-1,-1,-1],
                             [ 0, 0, 0],
                             [ 1, 1, 1]])

# Mask of the empty road
mask = cv2.imread("mask.jpg") # This image is of black and white pixels only
# The white pixels represent the region we need to consider for the traffic light

# Everything into single function
def process_the_frame(frame,gthreshold = 100):
        # Mask the frame with the mask to get the region of interest
        frame = cv2.bitwise_and(frame, mask)
        # 1. Convert the frame to grayscale
        gray = grayscale(frame)
        # 2. Apply Binary conversion
        binary = binary_conversion(gray, gthreshold)
        # 3. Apply Opening operation
        morphed_image = morph(binary, structure, "opening")
        # 4. Apply Prewitt edge detection
        prewitt_x = convolution(morphed_image, prewitt_kernel_x)
        prewitt_y = convolution(morphed_image, prewitt_kernel_y)
        prewitt = np.sqrt(prewitt_x**2 + prewitt_y**2)
        # Display Prewitt image
        # 5. Apply Blob analysis to count the number of blobs detected
        # and
        # 6. Draw bounding boxes around the blobs detected
        num_blobs,blob_image = blob_analysis(prewitt)

        other_images = [gray,binary,morphed_image,prewitt_x,prewitt_y,prewitt]
        return num_blobs,blob_image,other_images

# Open the video file
cap = cv2.VideoCapture('traffic_clip_2.mp4')

# Reference Image of Empty Road 
reference = cv2.imread("empty_road_downsized.jpg")

# Process the reference image
r_num, r_image, other = process_the_frame(reference)

# get an average value to use for binary conversion based on grayscale of reference image
grayscale_average = np.average(other[0])

# Set the tolerence value
tolerence = 20 # This value was taken arbitrarily
# Adjust the tolerence value to make the signal green only when there are a lot of vehicles
# The bigger the tolerence value, the more the number of vehicles required to make the signal green

# Set the reference values and image
reference_threshold = r_num + tolerence

# Processing and storing the reference image and the other images
cv2.imwrite(f"Processed/Reference/reference_threshold{str(reference_threshold)}.jpg", r_image)
other_types = ["gray","binary","morphed_image","prewitt_x","prewitt_y","prewitt"]
for i in range(len(other)):
    cv2.imwrite(f"Processed/Reference/reference_{other_types[i]}.jpg", other[i])
print("#Reference image processed.")
print("   >Reference threshold:", reference_threshold,"\n")
print("#Processing video...")

i = 0
# Read the video frame by frame
while(cap.isOpened()):
    # We will be using every 100th frame for processing
    ret, frame = cap.read()
    i += 1
    if i % 100 == 0:
        if ret == True:
            # Process the frame
            num_blobs, blob_image, _ = process_the_frame(frame,grayscale_average)
            # If the number of blobs cross a certain threshold make traffic light green
            threshold = reference_threshold
            if num_blobs > threshold:
                signal = "GREEN"
                print(f"   >Frame:{str(i)} Blobs:{num_blobs} Signal:Green")
            else:
                signal = "RED"
                print(f"   >Frame:{str(i)} Blobs:{num_blobs} Signal:Red")
            # Store the frame with bounding boxes in a folder called "Processed"
            cv2.imwrite(f"Processed/frame_{str(i)}_{signal}.jpg", blob_image)
        else:
            break

# Release the capture
cap.release()

print("\n#Done")
