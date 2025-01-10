import numpy as np
import nibabel as nib
import subprocess
import os
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter

def L_Phantom(phantomPath):
    # Create a 10x10x10 3D volume initialized with zeros
    volume = np.zeros((10, 10, 10), dtype=np.uint8)

    # Define the position of the "L" in the center of the volume
    z_slice = 5  # z index where the L will be placed

    # Vertical part of L: 7 rows in column 4
    volume[2:9, 4, z_slice] = 1  # L starts from row 2 to row 8 in column 4

    # Horizontal part of L: 4 columns in row 8
    volume[8, 4:8, z_slice] = 1  # L spans columns 4 to 7 at row 8

    volume[3:5, 5, 5] = 1

    volume[7, 4, 5] = 0

    # Create NIfTI image from the volume
    nii_img = nib.Nifti1Image(volume, np.eye(4))

    # Save the image to an NIfTI file
    nib.save(nii_img, phantomPath)
    print(f"NIfTI image saved as '{phantomPath}'")

def diagonal_branch_Phantom(phantomPath):
    # Create a 10x10x10 3D volume initialized with zeros
    volume = np.zeros((10, 10, 10), dtype=np.uint8)

    # Create a diagonal trunk (in x, y, z directions)
    # Moving diagonally from one corner of the cube to the opposite corner
    for i in range(3, 8):  # Diagonal main trunk
        volume[i, i, i] = 1

    # Diagonal branches at different points along the trunk
    # Branch 1 (diagonally upwards in the x-y plane at z = 4)
    for i in range(4, 7):
        volume[3, i, i] = 1  # Diagonal branch 1 along x-y plane at z = 4

    # Branch 2 (diagonal downwards, but at a gap)
    volume[7, 2, 7] = 1  # Diagonal branch 2 part 1
    volume[8, 3, 8] = 1  # Diagonal branch 2 part 2

    # Branch extending further into the z dimension
    for i in range(5, 8):
        volume[i, i - 1, i] = 1  # Diagonal extension of trunk

    # Create NIfTI image from the volume
    nii_img = nib.Nifti1Image(volume, np.eye(4))

    # Save the image to an NIfTI file
    nib.save(nii_img, phantomPath)
    print(f"NIfTI image saved as '{phantomPath}'")

def branch_Phantom(phantomPath):
    # Create a 10x10x10 3D volume initialized with zeros
    volume = np.zeros((10, 10, 10), dtype=np.uint8)

    # Create a branching tree structure with gaps
    z_slice = 5  # Central slice to build the main trunk of the tree

    # Main vertical trunk (with gaps)
    volume[2:5, 4, z_slice] = 1  # Trunk part 1
    volume[6:9, 4, z_slice] = 1  # Trunk part 2 (gap between rows 5 and 6)

    # Branches at different heights (with gaps in some)
    # Branch 1 at row 3
    volume[3, 4:7, z_slice] = 1  # First branch (left to right)
    
    # Branch 2 at row 7 (with a gap)
    volume[7, 2:4, z_slice] = 1  # Second branch (part 1)
    volume[7, 6:8, z_slice] = 1  # Second branch (part 2, with a gap between columns 4 and 6)

    # Extending the tree into the 3rd dimension with a branching structure
    volume[5, 4, 4:7] = 1  # Vertical extension in the z direction (central trunk)
    
    # Diagonal branches extending along z
    volume[3, 4, 6] = 1  # Diagonal branch 1
    volume[7, 4, 3] = 1  # Diagonal branch 2

    # Create NIfTI image from the volume
    nii_img = nib.Nifti1Image(volume, np.eye(4))

    # Save the image to an NIfTI file
    nib.save(nii_img, phantomPath)
    print(f"NIfTI image saved as '{phantomPath}'")

def cube_Phantom(phantomPath):
    # Create a 10x10x10 grid for the 3D cube
    volume = np.zeros((10, 10, 10), dtype=np.uint8)

    # Define the dimensions of the outer cube and hole
    cube_size = 6
    hole_size = cube_size - 2

    # volume[1:9, 6, 1:9] = 1 
    # volume[2:8,2:8,:] = 0
    # volume[2:8, :, 2:8] = 0
    # volume[:, 2:8, 2:8] = 0
    
    # Calculate the start and end indices for the cube to be centered in the volume
    start_index = (10 - cube_size) // 2
    end_index = start_index + cube_size

    # Create a 4x4x4 cube in the center of the volume
    volume[start_index:end_index, start_index:end_index, start_index:end_index] = 1  # outer cube

    lower = int(5 - (hole_size / 2))
    upper = int(5 + (hole_size / 2))

    # Apply the slicing with integer indices
    volume[lower:upper, lower:upper, :] = 0
    volume[lower:upper, :, lower:upper] = 0
    volume[:, lower:upper, lower:upper] = 0
    

    # Create a NIfTI image from the volume
    nii_img = nib.Nifti1Image(volume, np.eye(4))

    # Save the image to an NIfTI file
    nib.save(nii_img, phantomPath)
    print(f"NIfTI image saved as '{phantomPath}'")

def gradient(gradient):
    # The gradient data will have shape (X, Y, Z, 3), where the last dimension contains the gradient vectors
    # Normalize the gradient to get direction vectors (optional)
    vector_field = np.zeros_like(gradient)
    norm = np.linalg.norm(gradient, axis=-1, keepdims=True)
    
    # Avoid division by zero
    norm[norm == 0] = 1
    
    # Normalize to get unit vectors in the direction of the gradient
    vector_field = gradient / norm

    return vector_field

def PCA(image, kernel=3):
    x_dim, y_dim, z_dim = image.shape
    
    # Initialize the vector field with the same spatial dimensions, but with 3 components (for x, y, z directions)
    vector_field = np.zeros((x_dim, y_dim, z_dim, 3))
    
    # Define the radius of the neighborhood
    r = kernel // 2
    
    # Iterate over each voxel in the image (ignoring boundaries based on kernel size)
    for x in range(r, x_dim - r):
        for y in range(r, y_dim - r):
            for z in range(r, z_dim - r):
                if image[x,y,z]:
                    # Extract the local neighborhood (full 3D block)
                    neighborhood = image[x - r:x + r + 1, y - r:y + r + 1, z - r:z + r + 1]
                    
                    # Get the relative positions of non-zero intensity points in the neighborhood
                    points_relative = np.argwhere(neighborhood > 0)
                    
                    # If there are enough points to perform PCA
                    if len(points_relative) >= 3:
                        # Convert relative coordinates to full 3D coordinates in the image space
                        points_full = points_relative + np.array([x - r, y - r, z - r])  # Adjusting relative positions to the full image space
                        
                        # Perform PCA: mean-center the points
                        mean = np.mean(points_full, axis=0)
                        centered_points = points_full - mean

                        # Normalize by standard deviation along each axis to reduce axis skewness
                        std_devs = np.std(centered_points, axis=0)
                        centered_points /= (std_devs + 1e-6)  # Avoid division by zero

                        # Compute the covariance matrix of the 3D coordinates
                        covariance_matrix = np.cov(centered_points, rowvar=False)
                        
                        # Compute eigenvalues and eigenvectors
                        eigvals, eigvecs = np.linalg.eig(covariance_matrix)
                        
                        # Find the eigenvector corresponding to the largest eigenvalue (principal component)
                        principal_component = eigvecs[:, np.argmax(eigvals)]
                        
                        vector_field[x, y, z] = principal_component
                    else:
                        # If there are not enough points, set the vector to zero (no reliable orientation)
                        vector_field[x, y, z] = [0, 0, 0]
    return vector_field

def hessian(image, sigma):
    # Get dimensions of the image
    x_dim, y_dim, z_dim = image.shape

    # Initialize the vector field with the same spatial dimensions, but with 3 components (for x, y, z directions)
    vector_field = np.zeros((x_dim, y_dim, z_dim, 3))

    def compute_hessian(image, sigma):
        """
        Compute the Hessian matrix for each pixel in the image.
        Hessian components are second-order partial derivatives.
        """
        # Apply Gaussian smoothing
        smoothed_image = gaussian_filter(image, sigma)
        # smoothed_image = image
        # Compute second derivatives (Hessian components)
        Dxx = gaussian_filter(smoothed_image, sigma, order=(2, 0, 0))
        Dyy = gaussian_filter(smoothed_image, sigma, order=(0, 2, 0))
        Dzz = gaussian_filter(smoothed_image, sigma, order=(0, 0, 2))
        Dxy = gaussian_filter(smoothed_image, sigma, order=(1, 1, 0))
        Dxz = gaussian_filter(smoothed_image, sigma, order=(1, 0, 1))
        Dyz = gaussian_filter(smoothed_image, sigma, order=(0, 1, 1))
        
        return Dxx, Dyy, Dzz, Dxy, Dxz, Dyz
   
    # Compute Hessian matrix
    Dxx, Dyy, Dzz, Dxy, Dxz, Dyz = compute_hessian(image, sigma)
    
    # Eigenvalue decomposition of Hessian matrix
    for i in range(x_dim):
        for j in range(y_dim):
            for k in range(z_dim):
                H = np.array([[Dxx[i,j,k], Dxy[i,j,k], Dxz[i,j,k]],
                            [Dxy[i,j,k], Dyy[i,j,k], Dyz[i,j,k]],
                            [Dxz[i,j,k], Dyz[i,j,k], Dzz[i,j,k]]])
                
                # Eigenvalues of Hessian matrix
                eigvals, eigvecs = np.linalg.eigh(H)
                
                if image[i,j,k]:
                    vector_field[i,j,k] = eigvecs[:, np.argmin(abs(eigvals))] 
    return vector_field

# Generate the phantom
phantomPath = 'phantom.nii'
diagonal_branch_Phantom(phantomPath)

image_nii = nib.load(phantomPath)
image = image_nii.get_fdata()

# affine = image_nii.affine
affine = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])

image_dilated = gaussian_filter(image,1)

x_dim, y_dim, z_dim = image.shape

# Save PCA vector field
vector_field_PCA = PCA(image)
nifti_PCA = nib.Nifti1Image(vector_field_PCA, affine)
nib.save(nifti_PCA, 'vector_field_PCA.nii')

vector_field_hessian_5 = hessian(image, 1)
nifti_hessian_5 = nib.Nifti1Image(vector_field_hessian_5, affine)
nib.save(nifti_hessian_5, 'vector_field_hessian_5.nii')

vector_field_hessian_7 = hessian(image, 0.7)
nifti_hessian_7 = nib.Nifti1Image(vector_field_hessian_7, affine)
nib.save(nifti_hessian_7, 'vector_field_hessian_7.nii')

vector_field_hessian_10 = hessian(image, 10)
nifti_hessian_10 = nib.Nifti1Image(vector_field_hessian_10, affine)
nib.save(nifti_hessian_10, 'vector_field_hessian_10.nii')

# # Iterate over the image (excluding boundaries)
# for i in range(1, x_dim-1):
#     for j in range(1, y_dim-1):
#         for k in range(1, z_dim-1):
#             if image[i, j, k]:
#                 print(i,j,k,vector_field_hessian_5[i,j,k])
#                 print(i,j,k,vector_field_stickAverage[i,j,k])

image_dilated_nii = nib.Nifti1Image(image_dilated, affine)
nib.save(image_dilated_nii, 'phantom_dilated.nii')

# Command to visualize the result with mrview
mrview_command = f"""
    mrview -load {phantomPath} -mode 2 -voxel 7,7,7 -interpolation 0\
    -overlay.load phantom_dilated.nii -overlay.interpolation 0 -overlay.opacity 0.2\
    -fixel.load {'vector_field_hessian_7.nii'}
"""

subprocess.run(mrview_command, shell=True)